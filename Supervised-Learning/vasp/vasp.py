from utils import *
import tensorflow as tf
import tensorflow_addons as tfa

set_seed(42)

class DiagonalToZero(tf.keras.constraints.Constraint):
    def __call__(self, w):
        """Set diagonal to zero"""
        q = tf.linalg.set_diag(w, tf.zeros(w.shape[0:-1]), name=None)
        return q

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a basket."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), stddev=1.)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Model(tf.keras.Model):
    def __init__(self, num_words, latent=1024, hidden=1024, items_sampling=1.):
        """
        num_words             nr of items in dataset (size of tokenizer)
        latent                size of latent space
        hidden                size of hidden layers
        items_sampling        Large items datatsets can be very gpu memory consuming in EASE layer.
                              This coefficient reduces number of ease parametrs by taking only
                              fraction of items sorted by popularity as input for model.
                              Note: This coef should be somewhere around coverage@100 achieved by full
                              size model.
                              For ML20M this coef should be between 0.4888 (coverage@100 for full model)
                              and 1.0
                              For Netflix this coef should be between 0.7055 (coverage@100 for full
                              model) and 1.0
        """
        super(Model, self).__init__()

        self.sampled_items = int(num_words * items_sampling)

        assert self.sampled_items > 0
        assert self.sampled_items <= num_words

        self.s = self.sampled_items < num_words

        # ************* ENCODER ***********************
        self.encoder1 = tf.keras.layers.Dense(hidden)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.encoder2 = tf.keras.layers.Dense(hidden)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.encoder3 = tf.keras.layers.Dense(hidden)
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.encoder4 = tf.keras.layers.Dense(hidden)
        self.ln4 = tf.keras.layers.LayerNormalization()
        self.encoder5 = tf.keras.layers.Dense(hidden)
        self.ln5 = tf.keras.layers.LayerNormalization()
        self.encoder6 = tf.keras.layers.Dense(hidden)
        self.ln6 = tf.keras.layers.LayerNormalization()
        self.encoder7 = tf.keras.layers.Dense(hidden)
        self.ln7 = tf.keras.layers.LayerNormalization()

        # ************* SAMPLING **********************
        self.dense_mean = tf.keras.layers.Dense(latent,
                                                name="Mean")
        self.dense_log_var = tf.keras.layers.Dense(latent,
                                                   name="log_var")

        self.sampling = Sampling(name='Sampler')

        # ************* DECODER ***********************
        self.decoder1 = tf.keras.layers.Dense(hidden)
        self.dln1 = tf.keras.layers.LayerNormalization()
        self.decoder2 = tf.keras.layers.Dense(hidden)
        self.dln2 = tf.keras.layers.LayerNormalization()
        self.decoder3 = tf.keras.layers.Dense(hidden)
        self.dln3 = tf.keras.layers.LayerNormalization()
        self.decoder4 = tf.keras.layers.Dense(hidden)
        self.dln4 = tf.keras.layers.LayerNormalization()
        self.decoder5 = tf.keras.layers.Dense(hidden)
        self.dln5 = tf.keras.layers.LayerNormalization()

        self.decoder_resnet = tf.keras.layers.Dense(self.sampled_items,
                                                    activation='sigmoid',
                                                    name="DecoderR")
        self.decoder_latent = tf.keras.layers.Dense(self.sampled_items,
                                                    activation='sigmoid',
                                                    name="DecoderL")

        # ************* PARALLEL SHALLOW PATH *********

        self.ease = tf.keras.layers.Dense(
            self.sampled_items,
            activation='sigmoid',
            use_bias=False,
            kernel_constraint=DiagonalToZero(),  # critical to prevent learning simple identity
        )

    def call(self, x, training=None):
        sampling = self.s
        if sampling:
            sampled_x = x[:, :self.sampled_items]
            non_sampled = x[:, self.sampled_items:] * 0.
        else:
            sampled_x = x

        z_mean, z_log_var, z = self.encode(sampled_x)
        if training:
            d = self.decode(z)
            # Add KL divergence regularization loss.
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            self.add_loss(kl_loss)
            self.add_metric(kl_loss, name="kl_div")
        else:
            d = self.decode(z_mean)

        if sampling:
            d = tf.concat([d, non_sampled], axis=-1)

        ease = self.ease(sampled_x)

        if sampling:
            ease = tf.concat([ease, non_sampled], axis=-1)

        return d * ease

    def decode(self, x):
        e0 = x
        e1 = self.dln1(tf.keras.activations.swish(self.decoder1(e0)))
        e2 = self.dln2(tf.keras.activations.swish(self.decoder2(e1) + e1))
        e3 = self.dln3(tf.keras.activations.swish(self.decoder3(e2) + e1 + e2))
        e4 = self.dln4(tf.keras.activations.swish(self.decoder4(e3) + e1 + e2 + e3))
        e5 = self.dln5(tf.keras.activations.swish(self.decoder5(e4) + e1 + e2 + e3 + e4))

        dr = self.decoder_resnet(e5)
        dl = self.decoder_latent(x)

        return dr * dl

    def encode(self, x):
        e0 = x
        e1 = self.ln1(tf.keras.activations.swish(self.encoder1(e0)))
        e2 = self.ln2(tf.keras.activations.swish(self.encoder2(e1) + e1))
        e3 = self.ln3(tf.keras.activations.swish(self.encoder3(e2) + e1 + e2))
        e4 = self.ln4(tf.keras.activations.swish(self.encoder4(e3) + e1 + e2 + e3))
        e5 = self.ln5(tf.keras.activations.swish(self.encoder5(e4) + e1 + e2 + e3 + e4))
        e6 = self.ln6(tf.keras.activations.swish(self.encoder6(e5) + e1 + e2 + e3 + e4 + e5))
        e7 = self.ln7(tf.keras.activations.swish(self.encoder7(e6) + e1 + e2 + e3 + e4 + e5 + e6))

        z_mean = self.dense_mean(e7)
        z_log_var = self.dense_log_var(e7)
        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z

class VASP(tf.keras.Model):

    def __init__(self, dataset):
        super(VASP, self).__init__()
        self.dataset = dataset 
        #self.name = name
    
    def create_model(self, latent=2048, hidden=4096, ease_items_sampling=1., summary=False):
        self.model = Model(self.dataset.num_words, latent, hidden, ease_items_sampling)
        self.model(self.dataset.split.train_gen[0][0])
        if summary:
            self.model.summary()
        self.mc = MetricsCallback(self)

    def compile_model(self, lr=0.00002, fl_alpha=0.25, fl_gamma=2.0):
        """
        lr         learning rate of Nadam optimizer
        fl_alpha   alpha parameter of focal crossentropy
        fl_gamma   gamma parameter of focal crossentropy
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Nadam(lr),
            loss=lambda x, y: tfa.losses.sigmoid_focal_crossentropy(x, y, alpha=fl_alpha, gamma=fl_gamma),
            metrics=['mse', cosine_loss]
        )

    def train_model(self, epochs=150):
        self.model.fit(
            self.dataset.split.train_gen,
            validation_data=self.dataset.split.validation_gen,
            epochs=epochs,
            callbacks=[self.mc]
        )


dataset = Data(d='', pruning='u5')
dataset.splits = []
dataset.create_splits(1, 10000, shuffle=False, generators=False)
dataset.split.train_users = pd.read_json("../../data/train_users.json").userid.apply(str).to_frame()
dataset.split.validation_users = pd.read_json("../../data/val_users.json").userid.apply(str).to_frame()
dataset.split.test_users = pd.read_json("../../data/test_users.json").userid.apply(str).to_frame()
dataset.split.generators()


#m = VASP(dataset.split, name="VASP_ML20_1")
m = VASP(dataset)
m.create_model(latent=2048, hidden=4096, ease_items_sampling=0.33)
m.model.summary()
print("=" * 80)
print("Train for 50 epochs with lr 0.00005")
m.compile_model(lr=0.00005, fl_alpha=0.25, fl_gamma=2.0)
m.train_model(50)
print("=" * 80)
print("Than train for 20 epochs with lr 0.00001")
m.compile_model(lr=0.00001, fl_alpha=0.25, fl_gamma=2.0)
m.train_model(20)
print("=" * 80)
print("Than train for 20 epochs with lr 0.000001")
m.compile_model(lr=0.00001, fl_alpha=0.25, fl_gamma=2.0)
m.train_model(20)
path="results.png"
m.mc.plot_history(path=path)

test_r20s = []
test_r50s = []
test_n100s = []

for fold in range(1,6):
    ev=Evaluator(m.split, method=str(fold)+'_20')
    ev.update(m.model)

    test_n100s.append(ev.get_ncdg(100))
    test_r20s.append(ev.get_recall(20))
    test_r50s.append(ev.get_recall(50))

print("TEST SET (MEAN)")
print("5-fold mean NCDG@100", round(sum(test_n100s) / len(test_n100s),3))
print("5-fold mean Recall@20", round(sum(test_r20s) / len(test_r20s),3))
print("5-fold mean Recall@50", round(sum(test_r50s) / len(test_r50s),3))
