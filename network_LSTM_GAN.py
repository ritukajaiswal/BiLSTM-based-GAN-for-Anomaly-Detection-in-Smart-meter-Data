from __future__ import print_function, division

from keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, RepeatVector, TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model

def get_generator_LSTM(optimizer):
    generator = Sequential()
    generator.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(8,1)))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Bidirectional(LSTM(128, return_sequences=True)))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Bidirectional(LSTM(128)))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(RepeatVector(8))
    generator.add(Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2)))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2)))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2)))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Dropout(0.3))   
    generator.add(TimeDistributed(Dense(128)))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Dropout(0.4))
    generator.add(TimeDistributed(Dense(128)))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Dropout(0.4))
    generator.add(TimeDistributed(Dense(1)))
    generator.add(LeakyReLU(alpha=0.2))
    # generator.summary()

    noise = Input(shape=(8,1))
    img = generator(noise)

    return Model(noise, img)

    # generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # return generator

def get_discriminator_LSTM(optimizer):
    discriminator = Sequential()
    discriminator.add(Bidirectional(LSTM(128, activation = 'relu', return_sequences=True), input_shape=(8,1)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Bidirectional(LSTM(128, activation = 'relu')))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.4))
    discriminator.add(RepeatVector(1))
    discriminator.add(TimeDistributed(Dense(128, activation = 'sigmoid')))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.4))
    discriminator.add(TimeDistributed(Dense(128, activation = 'relu')))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.4))
    discriminator.add(TimeDistributed(Dense(1, activation = 'linear')))
    # discriminator.summary()

    img = Input(shape=(8,1))
    validity = discriminator(img)

    return Model(img, validity)

    # discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # return discriminator


def get_gan_network_LSTM(discriminator, generator, optimizer, input_dim=8):
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    z = Input(shape=(8,1))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)
    # gan_input = Input(shape=(input_dim,1))
    # x = generator(gan_input)
    # gan_output = discriminator(generator)

    gan = Model(z, valid)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    return gan