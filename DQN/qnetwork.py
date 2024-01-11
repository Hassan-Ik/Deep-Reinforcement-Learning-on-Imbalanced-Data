import tensorflow as tf

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate: float, image:bool = True):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.image = image
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        
        # Main Q-network
        self.model = self.build_network(self.image)

    def build_network(self, image=True):

        if image == True:
            # Input layer of the network
            input_layer = tf.keras.layers.Input(shape = self.state_size)

            # Convolution Layers
            conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation=tf.nn.relu)(input_layer)
            conv2 = tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu)(conv2)
            maxpooling = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv2)
            
            flatten = tf.keras.layers.Flatten()(maxpooling)
            dense = tf.keras.layers.Dense(64, activation='relu')(flatten)
            outputs = tf.keras.layers.Dense(self.action_size, activation="softmax")(dense)
            model = tf.keras.Model(inputs=input_layer, outputs=outputs)
            
            # Compiling Deep Q Network Model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            
            return model
        else:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=self.state_size),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.action_size, activation='softmax')  # Output layer for binary classification
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()
            return model

class DDQNetwork:
    def __init__(self, state_size, action_size, learning_rate: float, image:bool = True):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.image = image
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        
        # Main Q-network
        self.model = self.build_network(self.image)

        # Target Q-network
        self.target_model = self.build_network(self.image)
        
        self.update_target_network()
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def build_network(self, image=True):

        if image == True:
            # Input layer of the network
            input_layer = tf.keras.layers.Input(shape = self.state_size)

            # Convolution Layers
            conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation=tf.nn.relu)(input_layer)
            conv2 = tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu)(conv2)
            maxpooling = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv2)
            
            flatten = tf.keras.layers.Flatten()(maxpooling)
            dense = tf.keras.layers.Dense(64, activation='relu')(flatten)
            outputs = tf.keras.layers.Dense(self.action_size, activation="softmax")(dense)
            model = tf.keras.Model(inputs=input_layer, outputs=outputs)
            
            # Compiling Deep Q Network Model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            
            return model
        else:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=self.state_size),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(self.action_size, activation='softmax')  # Output layer for binary classification
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()
            return model