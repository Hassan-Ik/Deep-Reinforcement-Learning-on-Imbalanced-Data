import tensorflow as tf

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate: float):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        
        # Main Q-network
        self.model = self.build_network()

        # Target Q-network
        self.target_model = self.build_network()
        
        self.update_target_network()
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def build_network(self):
        # Input layer of the network
        input_layer = tf.keras.layers.Input(shape = self.state_size)

        # Convolution Layers
        conv2 = tf.keras.layers.Conv2D(32, 5, strides=2, activation=tf.nn.relu)(input_layer)
        conv2 = tf.keras.layers.Conv2D(32, 5, strides=2, activation=tf.nn.relu)(conv2)
        flatten = tf.keras.layers.Flatten()(conv2)
        outputs = tf.keras.layers.Dense(self.action_size, activation="softmax")(flatten)
        model = tf.keras.Model(inputs=input_layer, outputs=outputs)
        
        # Compiling Deep Q Network Model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        
        return model