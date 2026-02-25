import tensorflow as tf 
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

Model   = tf.keras.Model
Input   = tf.keras.Input
Dense   = tf.keras.layers.Dense
Dropout = tf.keras.Dropout

# Build a Multi-Layer Perceptron with Keras functional API

def build_mlp(input_dim, hidden_layers, output_units, output_activation, dropout_rate):
    #Create input layer with input dimesion 'input_dim'
    inputs = Input(shape=(input_dim,))
    x = inputs
    
    # Add hidden layers recursively
    for units, activation in hidden_layers:
        x = Dense(units, activation=activation)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # Create Output layer with output dimension 'output_units'
    outputs = Dense(output_units, activation=output_activation)(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model