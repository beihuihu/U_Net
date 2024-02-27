#   Author: Ankit Kariryaa, University of Bremen

from tensorflow.keras import models, layers
from tensorflow.keras import regularizers

def UNet(input_shape,input_label_channel, layer_count=64, regularizers = regularizers.l2(0.0001), gaussian_noise=0.1, weight_file = None):
        """ Method to declare the UNet model.
        Args:  
            input_shape: tuple(int, int, int, int)     
                Shape of the input in the format (batch, height, width, channels).
            input_label_channel: list([int])  
                list of index of label channels, used for calculating the number of channels in model output.
            layer_count: (int, optional) 
                Count of kernels in first layer. Number of kernels in other layers grows with a fixed factor.
            regularizers: keras.regularizers 
                regularizers to use in each layer.
            weight_file: str  
                path to the weight file.
        """
        input_img = layers.Input(input_shape[1:], name='Input')
        pp_in_layer  = input_img
#        pp_in_layer = layers.GaussianNoise(gaussian_noise)(input_img)
#        pp_in_layer = layers.BatchNormalization()(pp_in_layer)

        c1 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='valid')(pp_in_layer)
        c1 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='valid')(c1)
        n1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D((2, 2))(n1)

        c2 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='valid')(p1)
        c2 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='valid')(c2)
        n2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(n2)

        c3 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='valid')(p2)
        c3 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='valid')(c3)
        n3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(n3)

        c4 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='valid')(p3)
        c4 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='valid')(c4)
        n4 = layers.BatchNormalization()(c4)
        # drop4 = Dropout(0.5)(n4) 
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(n4)

        c5 = layers.Conv2D(16*layer_count, (3, 3), activation='relu', padding='valid')(p4)
        c5 = layers.Conv2D(16*layer_count, (3, 3), activation='relu', padding='valid')(c5)

        u6 = layers.UpSampling2D((2, 2))(c5)
        n6 = layers.BatchNormalization()(u6)
        deltah = (n4.get_shape().as_list()[1]-n6.get_shape().as_list()[1])/2
        deltaw = (n4.get_shape().as_list()[2]-n6.get_shape().as_list()[2])/2
        u6 = layers.concatenate([n6, layers.Cropping2D(cropping=(int(deltah),int(deltaw)))(n4)])
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='valid')(u6)
        c6 = layers.Conv2D(8*layer_count, (3, 3), activation='relu', padding='valid')(c6)

        u7 = layers.UpSampling2D((2, 2))(c6)
        n7 = layers.BatchNormalization()(u7)
        deltah = (n3.get_shape().as_list()[1]-n7.get_shape().as_list()[1])/2
        deltaw = (n3.get_shape().as_list()[2]-n7.get_shape().as_list()[2])/2
        u7 = layers.concatenate([n7, layers.Cropping2D(cropping=(int(deltah),int(deltaw)))(n3)])
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='valid')(u7)
        c7 = layers.Conv2D(4*layer_count, (3, 3), activation='relu', padding='valid')(c7)

        u8 = layers.UpSampling2D((2, 2))(c7)
        n8 = layers.BatchNormalization()(u8)
        deltah = (n2.get_shape().as_list()[1]-n8.get_shape().as_list()[1])/2
        deltaw = (n2.get_shape().as_list()[2]-n8.get_shape().as_list()[2])/2
        u8 = layers.concatenate([n8, layers.Cropping2D(cropping=(int(deltah),int(deltaw)))(n2)])
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='valid')(u8)
        c8 = layers.Conv2D(2*layer_count, (3, 3), activation='relu', padding='valid')(c8)

        u9 = layers.UpSampling2D((2, 2))(c8)
        n9 = layers.BatchNormalization()(u9)
        deltah = (n1.get_shape().as_list()[1]-n9.get_shape().as_list()[1])/2
        deltaw = (n1.get_shape().as_list()[2]-n9.get_shape().as_list()[2])/2
        u9 = layers.concatenate([n9, layers.Cropping2D(cropping=(int(deltah),int(deltaw)))(n1)], axis=3)
        c9 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='valid')(u9)
        c9 = layers.Conv2D(1*layer_count, (3, 3), activation='relu', padding='valid')(c9)

#         d = layers.Conv2D(len(input_label_channel), (1, 1), activation='sigmoid', kernel_regularizer= regularizers)(c9)
        x = layers.Conv2D(len(input_label_channel), (1, 1), kernel_regularizer= regularizers)(c9)
        d = layers.Activation('sigmoid', dtype='float32')(x)
        seg_model = models.Model(inputs=[input_img], outputs=[d])
        if weight_file:
            seg_model.load_weights(weight_file)
        seg_model.summary()
        return seg_model
