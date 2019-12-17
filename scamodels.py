import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Encoder(nn.Module):
    """
    Encoder.
    Ouput only feature maps
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class SpatialAttention(nn.Module):
    """
    SpatialAttention Network.
    """

    def __init__(self, encoder_shape, decoder_dim, k):
        """
        :param encoder_shape: feature map size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param k: size of the transform matrice
        """
        super(SpatialAttention, self).__init__()
        _,C,H,W = tuple([int(x) for x in encoder_shape])
        self.W_s = nn.Parameter(torch.randn(C,k))
        self.W_hs = nn.Parameter(torch.randn(k,decoder_dim))
        self.W_i = nn.Parameter(torch.randn(k,1))
        self.b_s = nn.Parameter(torch.randn(k))
        self.b_i = nn.Parameter(torch.randn(1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: feature map, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, alpha
        """
        V = encoder_out.view(encoder_out.shape[0],2048,-1).permute(0,2,1) # vector fot each channel
        att = self.tanh((torch.matmul(V, self.W_s)+self.b_s)+(torch.matmul(decoder_hidden,self.W_hs).unsqueeze(1)))  # (batch_size, num_pixels, channel)
        alpha = self.softmax(torch.matmul(att,self.W_i) + self.b_i).squeeze(2)  # (batch_size, num_pixels)
        encoder_new = encoder_out.view(encoder_out.shape[0],2048,-1)
        attention_weighted_encoding = torch.mul(encoder_new, alpha.unsqueeze(1))  # (batch_size, encoder_shape)
        return attention_weighted_encoding, alpha


class ChannelWiseAttention(nn.Module):
    """
    ChannelWiseAttention Network.
    """

    def __init__(self, encoder_shape, decoder_dim, k):
        """
        :param encoder_shape: feature map size of encoded images
        :param decoder_dim: size of decoder's RNN
        """
        super(ChannelWiseAttention, self).__init__()
        _,C,H,W = tuple([int(x) for x in encoder_shape])
        self.W_c = nn.Parameter(torch.randn(1,k))
        self.W_hc = nn.Parameter(torch.randn(k,decoder_dim))
        self.W_i_hat = nn.Parameter(torch.randn(k,1))
        self.b_c = nn.Parameter(torch.randn(k))
        self.b_i_hat = nn.Parameter(torch.randn(1))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 0)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: feature map, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, beta
        """
        V = encoder_out.view(encoder_out.shape[0],2048,-1).mean(dim=2).unsqueeze(2) # (batch_size, channel, 1)
        att = self.tanh((torch.matmul(V, self.W_c)+self.b_c)+(torch.matmul(decoder_hidden,self.W_hc).unsqueeze(1)))  # (batch_size, channnel, k)
        beta = self.softmax(torch.matmul(att,self.W_i_hat) + self.b_i_hat).unsqueeze(2)  # (batch_size, channel, 1)
        attention_weighted_encoding = torch.mul(encoder_out, beta)  # (batch_size, encoder_shape)
        return attention_weighted_encoding, beta




class DecoderWithSCACNNAttention(nn.Module):
    """
    DecoderWithSCACNNAttention
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, encoder_shape=[1,2048,8,8], k=512, dropout=0.5, disabled=False):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_shape: feature size of encoded images
        :param k: size of the transform matrice
        :param dropout: dropout
        """
        super(DecoderWithSCACNNAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.AvgPool = nn.AvgPool2d(8)

        self.SpatialAttention  = SpatialAttention(encoder_shape, decoder_dim, k)  # SpatialAttention network
        self.ChannelWiseAttention  = ChannelWiseAttention(encoder_shape, decoder_dim, k)  # ChannelWiseAttention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = self.AvgPool(encoder_out).squeeze(-1).squeeze(-1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size


        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, beta = self.ChannelWiseAttention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            attention_weighted_encoding, alpha = self.SpatialAttention(attention_weighted_encoding[:batch_size_t],
                                                                h[:batch_size_t])
            attention_weighted_encoding = attention_weighted_encoding.view(attention_weighted_encoding.shape[0],2048,8,8)
            attention_weighted_encoding = self.AvgPool(attention_weighted_encoding)
            attention_weighted_encoding = attention_weighted_encoding.squeeze(-1).squeeze(-1) # decrease dimension
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds


        #To align with normal attention
        num_pixels = encoder_out.size(1)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
