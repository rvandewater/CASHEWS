import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# import sparselinear as sl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_hidden(
    x: torch.Tensor, hidden_size: int, bilstm: bool, input_size: float, xavier: bool = True
) -> torch.Tensor:
    """Creates the initial hidden und cell states of an LSTM cell

    Parameters
    ----------
    x : torch.Tensor
        Input data into the model
    hidden_size : int
        Size of the hidden layer
    bilstm : bool
        Wether to use bidirectional LSTM Cells
    input_size : float
        Size of the input layer
    xavier : bool, optional
        Using xavier initalization, by default True

    Returns
    -------
    torch.Tensor | nn.Variable
        returns a tensor intlalized with xavier normal, or a nn.Variable Object with a tensor
    """

    if xavier:
        if bilstm:
            return nn.init.xavier_normal_(
                torch.zeros(input_size * 2, x.size(0), hidden_size, dtype=torch.float)
            ).to(DEVICE)
        else:
            return nn.init.xavier_normal_(
                torch.zeros(input_size, x.size(0), hidden_size, dtype=torch.float)
            ).to(DEVICE)

    else:
        if bilstm:
            return torch.zeros(input_size * 2, x.size(0), hidden_size, dtype=torch.float).to(DEVICE)

        else:
            torch.zeros(input_size, x.size(0), hidden_size, dtype=torch.float).to(DEVICE)


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        attention: bool = True,
        attention_dropout: float | None = 0.2,
        num_layers: int | None = 1,
        bilstm: bool = True,
        num_heads: float | None = 4,
        sparse: bool | None = True,
    ):
        """Constructor of the LSTMENCODER class: genrates the encoder of the autoencoder. It is up to the user wether to use attention and/or bi-directional LSTM layers. The attention mechanism can be customized by wether using the sparse implementation and more randomness

        Parameters
        ----------
        input_size : int
            size of the input layer
        hidden_size : int
            size of the hidden layer
        dropout : float, optional
            Value of how percent should be masked, the value needs to be between (0.0,1.0], by default 0.1
        attention : bool, optional
            To use attention in the encoder, set this value to true, by default True
        bilstm : bool, optional
            To use the bidirectional lstm layers set to true, by default True
        num_heads : Optional[float], optional
            number of heads for the multi-head attention, should be passed when attention=true. In addition num_heads =1 equals Single Attention, WARNING: Model Dimension / Number of Heads %0, a higher number of heads leads to a poor performance, by default 4
        sparse : Optional[bool], optional
            To use the sparse attention implementation for saving space, time and money, by default True
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.attention = attention
        self.bilstm = bilstm
        self.dropout = dropout
        self.num_layers = num_layers

        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bilstm,
            dropout=self.dropout,
            device=DEVICE,
        )

        if attention:
            if bilstm:
                self.attn = MultiHeadAttention(
                    dim_model=self.hidden_size * 2,
                    num_heads=num_heads,
                    dropout=attention_dropout,
                    sparse=sparse,
                )
            else:
                self.attn = MultiHeadAttention(
                    dim_model=self.hidden_size, num_heads=num_heads, dropout=attention_dropout, sparse=sparse
                )

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:

        hidden_state = init_hidden(in_data, self.hidden_size, self.bilstm, self.num_layers)
        cell_state = init_hidden(in_data, self.hidden_size, self.bilstm, self.num_layers)
        in_data = in_data.permute(0, 2, 1)

        # LSTM Shape (Batch_size, sequence_length, input_features)
        hs, cs = self.LSTM(in_data, (hidden_state, cell_state))

        if self.attention:
            # (batch_size, sequence_length, input_features)
            hs = self.attn(hs, hs, hs)

        return hs

    def get_embedding(self, in_data: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding of the input data.

        Args:
            in_data: (torch.Tensor): tensor of input data
        """
        enc_out = self.forward(in_data)
        return enc_out


class LSTMDecoder(nn.Module):

    def __init__(
        self,
        hidden_size_encoder: int,
        hidden_size: int,
        output_size: int,
        bilstm: bool,
        dropout: float = 0.1,
        num_layers: int | None = 1,
    ):
        """Constructor of the LSTMDECODER class.

        Parameters
        ----------
        hidden_size_encoder : int
            Size of the input = size of the output of the LSTMENCODER
        hidden_size : int
            Size of the hidden layer
        output_size : int
            number of classes, if it is a regression task the number of classes should be 1
        bilstm : bool
            If bi-directional layers were used in the encoder
        dropout : float, optional
            value of how much should be masked (choose a value between (0.0, 1.0]), by default 0.1
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size = hidden_size
        self.bilstm = bilstm
        self.dropout = dropout
        self.num_layers = num_layers

        self.LSTM_dec = nn.LSTM(
            input_size=self.hidden_size_encoder * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bilstm,
            dropout=self.dropout,
            device=DEVICE,
        )
        if bilstm:
            self.fc = nn.Linear(in_features=self.hidden_size * 2, out_features=output_size, device=DEVICE)
        else:
            self.fc = nn.Linear(in_features=self.hidden_size, out_features=output_size, device=DEVICE)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, encoded_input: torch.Tensor) -> torch.Tensor:
        """Forward call of the LSTMDECODER, sends the data trough the LSTM Decoder.

        Parameters
        ----------
        encoded_input : torch.Tensor
            Output of the LSTMENCODER

        Returns
        -------
        torch.Tensor
            output of the autoencoder
        """
        hidden_state = init_hidden(encoded_input, self.hidden_size, self.bilstm, self.num_layers)
        cell_state = init_hidden(encoded_input, self.hidden_size, self.bilstm, self.num_layers)

        hs, cs = self.LSTM_dec(encoded_input, (hidden_state, cell_state))

        return self.fc(hs)


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_model: float, num_heads: float, dropout: float, sparse: bool):
        """Multi-Head Attention: Implementation of the single dot attention for the multi-head attention. With a sparse implementation to save ressources and the space, with Dropout to add some randomness to the attention weights

        Parameters
        ----------
        dim_model : float
            Dimension of the model
        num_heads : float
            Number of heads, the dimension of the models should be divideable by the number of heads. A higher number of heads might lead to a poor performance!
        dropout : float
            Amount of values which should be zeroed out
        sparse : bool
            To use the sparse implementation set it to True.
        """
        super().__init__()
        assert (
            dim_model % num_heads == 0
        ), "The dimension of the models should be dividable by the numbers of heads."

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_k = dim_model // num_heads

        if sparse:
            # Not implemented yet
            self.W_q = nn.Linear(dim_model, dim_model, device=DEVICE)
            self.W_k = nn.Linear(dim_model, dim_model, device=DEVICE)
            self.W_v = nn.Linear(dim_model, dim_model, device=DEVICE)
            # import sparselinear as sl
            # self.W_q = sl.SparseLinear(dim_model, dim_model)
            # self.W_k = sl.SparseLinear(dim_model, dim_model)
            # self.W_v = sl.SparseLinear(dim_model, dim_model)

        else:
            self.W_q = nn.Linear(dim_model, dim_model, device=DEVICE)
            self.W_k = nn.Linear(dim_model, dim_model, device=DEVICE)
            self.W_v = nn.Linear(dim_model, dim_model, device=DEVICE)

        self.dropout = dropout
        self.Dropout = nn.Dropout(dropout)
        self.W_o = nn.Linear(dim_model, dim_model)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Scaled dot attention: Is the matrix muliplication between the query and the key, where the result is scaled by the square root of the key dimension. The scaled result is multiplied by the value.

        Parameters
        ----------
        query : torch.Tensor
            Query vector
        key : torch.Tensor
            Key vector
        value : torch.Tensor
            Value vector
        mask : Optional[torch.BoolTensor], optional
            boolean mask, by default None

        Returns
        -------
        torch.Tensor
            attention probabilities
        """
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, value)
        return output

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the data in order to apply the attention per head

        Parameters
        ----------
        x : torch.Tensor
            encoded lstm output

        Returns
        -------
        torch.Tensor
            tensor made ready for the heads

        """
        batch_size, seq_length, dim_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.dim_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combining the output of the heads

        Parameters
        ----------
        x : torch.Tensor
            output of the attention mechanism

        Returns
        -------
        torch.Tensor
            combinded output of the attention mechanism
        """
        batch_size, num_heads, seq_length, dim_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.dim_model)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: None | None = None
    ) -> torch.Tensor:
        """Foward call of the multi head attention. If query=key=value: Self.Attention

        Parameters
        ----------
        query : torch.Tensor
            Query tensor
        key : torch.Tensor
            Key
        value : torch.Tensor
            Value
        mask : Optional[None], optional
            mask for the scaled dot attention, by default None

        Returns
        -------
        torch.Tensor
            output of the attention mechanism
        """
        query = self.split_heads(self.Dropout(self.W_q(query)))
        key = self.split_heads(self.Dropout(self.W_k(key)))
        value = self.split_heads(self.Dropout(self.W_v(value)))

        attn_output = self.scaled_dot_product_attention(query, key, value, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class LAAE(nn.Module):
    def __init__(self, encoder: LSTMEncoder, decoder: LSTMDecoder):
        """Constructor of the LAAE class

        Parameters
        ----------
        encoder : LSTMENCODER
            Encoder of the autoencoder
        decoder : LSTMDECODER
            Decoder of the autoencoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, time_series: torch.Tensor) -> torch.Tensor:
        """Forward call: Runs the autoencoder.

        Parameters
        ----------
        time_series : torch.Tensor
            Input data, a multi-modale time series

        Returns
        -------
        torch.Tensor
            Predicted values
        """
        # time_series = time_series.permute(0,2,1)

        encoder_output = self.encoder(time_series)
        decoder_output = self.decoder(encoder_output)
        decoder_output = decoder_output.permute(0, 2, 1)
        # print(decoder_output.shape)

        return decoder_output
