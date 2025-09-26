import torch
from pytorch_tcn import TCN
from torch import nn


class TCNAE_Encoder(nn.Module):
    def __init__(
        self,
        ts_dimension=1,
        dilations=(1, 2, 4, 8, 16),
        nb_filters=20,
        kernel_size=20,
        # nb_stacks=1,
        padding="same",
        dropout_rate=0.00,
        filters_conv1d=8,
        activation_conv1d="linear",
        latent_sample_rate=32,
        pooler=nn.AvgPool1d,
        conv_kernel_init="xavier_normal",
        use_norm="weight_norm",
    ):
        """
        Parameters
        ----------
        ts_dimension : int
            The dimension of the time series (default is 1)
        dilations : tuple
            The dilation rates used in the TCN-AE model (default is (1, 2, 4, 8, 16))
        nb_filters : int
            The number of filters used in the dilated convolutional layers. All dilated conv. layers use the same number of filters (default is 20)
        """
        super().__init__()
        self.ts_dimension = ts_dimension
        self.dilations = dilations
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        # self.nb_stacks = nb_stacks
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.filters_conv1d = filters_conv1d
        self.activation_conv1d = activation_conv1d  # todo implement actual nn.Function
        self.latent_sample_rate = latent_sample_rate
        self.conv_kernel_init = conv_kernel_init
        self.use_norm = use_norm

        # build the model

        # Put signal through TCN. Input-shape:  (batch_size, input_dim, sequence length),
        #                         Output-shape: (batch_size, nb_filters, sequence length)
        self.enc_tcn = TCN(
            num_inputs=self.ts_dimension,
            num_channels=[self.nb_filters] * len(self.dilations),
            kernel_size=self.kernel_size,
            # nb_stacks=self.nb_stacks,
            dilations=self.dilations,
            causal=True,
            use_skip_connections=True,
            dropout=self.dropout_rate,
            # return_sequences=True,
            kernel_initializer=self.conv_kernel_init,
            use_norm=self.use_norm,
            input_shape="NCL",  # batch_size, channels, sequence_length
        )

        # Now, adjust the number of channels...
        self.enc_flat = nn.Conv1d(
            in_channels=self.nb_filters, out_channels=self.filters_conv1d, kernel_size=1, padding=self.padding
        )

        self.enc_pooled = pooler

        # If you want, maybe put the pooled values through a non-linear Activation
        self.enc_out = self.enc_pooled

        self.flatten = nn.Flatten()

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        """
        Run forward computation.

        Args:
            in_data: (torch.Tensor): tensor of input data
        """
        enc_tcn = self.enc_tcn(in_data)
        enc_pooled = self.enc_pooled(enc_tcn)
        enc_out = self.enc_flat(enc_pooled)
        if self.activation_conv1d != "linear":
            enc_out = self.activation_conv1d(enc_out)

        return enc_out

    def get_embedding(self, in_data: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding of the input data.

        Args:
            in_data: (torch.Tensor): tensor of input data
        """
        enc_out = self.forward(in_data)
        return self.flatten(enc_out)


class TCNAE_Decoder(nn.Module):
    def __init__(
        self,
        ts_dimension=1,
        dilations=(1, 2, 4, 8, 16),
        nb_filters=20,
        kernel_size=20,
        # nb_stacks=1,
        dropout_rate=0.00,
        filters_conv1d=8,
        latent_sample_rate=32,
        conv_kernel_init="xavier_normal",
        use_norm="weight_norm",
    ):
        super().__init__()
        self.ts_dimension = ts_dimension
        self.dilations = dilations
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        # self.nb_stacks = nb_stacks
        self.dropout_rate = dropout_rate
        self.filters_conv1d = filters_conv1d
        self.latent_sample_rate = latent_sample_rate
        self.conv_kernel_init = conv_kernel_init
        self.use_norm = use_norm

        # build the model
        sampling_factor = self.latent_sample_rate

        # Now we should have a short sequence, which we will upsample again and then try to reconstruct the original series
        self.dec_upsample = nn.Upsample(scale_factor=sampling_factor)

        self.dec_tcn = TCN(
            num_inputs=filters_conv1d,
            num_channels=[self.nb_filters] * len(self.dilations),
            kernel_size=self.kernel_size,
            # nb_stacks=self.nb_stacks,
            dilations=self.dilations,
            causal=True,
            use_skip_connections=True,
            dropout=self.dropout_rate,
            # return_sequences=True,
            kernel_initializer=self.conv_kernel_init,
            use_norm=self.use_norm,
            input_shape="NCL",  # batch_size, channels, sequence_length
        )

        # Put the filter-outputs through a dense layer finally, to get the reconstructed signal
        self.dec_out = nn.Linear(self.nb_filters, self.ts_dimension)

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        upsampled = self.dec_upsample(in_data)
        dec_tcn = self.dec_tcn(upsampled)
        dec_tcn = torch.transpose(dec_tcn, 1, 2)
        dec_out = self.dec_out(dec_tcn)
        dec_out = torch.transpose(dec_out, 1, 2)

        return dec_out


class TCNAE(nn.Module):
    """
    A class used to represent the Temporal Convolutional Autoencoder (TCN-AE).

    ...

    Attributes
    ----------
    model : xxtypexx
        The TCN-AE model.

    Methods
    -------
    build_model(verbose = 1)
        Builds the model
    """

    def __init__(
        self,
        ts_dimension=1,
        dilations=(1, 2, 4, 8, 16),
        nb_filters=20,  # 1
        kernel_size=20,
        # nb_stacks=1,
        padding="same",
        dropout_rate=0.00,
        filters_conv1d=8,
        activation_conv1d="linear",
        latent_sample_rate=32,
        pooler=nn.AvgPool1d,
        conv_kernel_init="xavier_normal",
        use_norm="weight_norm",
    ):
        super().__init__()
        self.encoder = TCNAE_Encoder(
            ts_dimension=ts_dimension,
            dilations=dilations,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            # nb_stacks=nb_stacks,
            padding=padding,
            dropout_rate=dropout_rate,
            filters_conv1d=filters_conv1d,
            activation_conv1d=activation_conv1d,
            latent_sample_rate=latent_sample_rate,
            pooler=pooler,
            conv_kernel_init=conv_kernel_init,
            use_norm=use_norm,
        )
        self.decoder = TCNAE_Decoder(
            ts_dimension=ts_dimension,
            dilations=dilations,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            # nb_stacks=nb_stacks,
            dropout_rate=dropout_rate,
            filters_conv1d=filters_conv1d,
            latent_sample_rate=latent_sample_rate,
            conv_kernel_init=conv_kernel_init,
            use_norm=use_norm,
        )

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(in_data)
        dec_out = self.decoder(enc_out)

        return dec_out
