import vgg16
import load_dataset_simulator
import load_dataset_spurv
import save_load_model
#import heat_map
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import CSVLogger

import numpy as np
import math
import matplotlib.pyplot as plt



def main():

    #print("Creating model...")
    #vgg16.vgg16()

    #print("Loading model...")
    #model = load_model("vgg16_test.h5")
    #model = load_model("trained_model_simulator.h5")
    #model = load_model("model-010.h5")

    #csv_logger = CSVLogger('log_wednesday.csv', append=True, separator=';')
    #checkpoint = ModelCheckpoint('curr_best_model.h5', monitor='val_loss',verbose=0,save_best_only=True, mode='auto') #Saved_models


    print("Loading datasets...")
    np_images, np_steering = load_dataset_simulator.load_dataset("center","LEFT")
    
    #np_images_2, np_steering_2 = load_dataset_simulator.load_dataset("center","mond2")
    
    #np_images_3, np_steering_3 = load_dataset_simulator.load_dataset("center","track1_rewind")

    #np_val_images, np_val_steering = load_dataset_simulator.load_dataset("center","test")
    #np_val_images_new, np_val_steering_new = load_dataset_simulator.load_dataset("right","test")

    #np_val_images = np.concatenate((np_val_images, np_val_images_new))
    #np_val_steering = np.concatenate((np_val_steering, np_val_steering_new))
    
    #np_val_images_new, np_val_steering_new = load_dataset_simulator.load_dataset("left","test")

    #np_val_images = np.concatenate((np_val_images, np_val_images_new))
    #np_val_steering = np.concatenate((np_val_steering, np_val_steering_new))
    
    #print("Length of val images: ", len(np_val_images))
    #np.save("np_images",np_val_images)

    #print("Length of val steer: ", len(np_val_steering))
    #np.save("np_steering",np_val_steering)

    use_2 = False
    use_3 = False

    for dataset in ["LEFT", "RIGHT", "mond", "mond2", "mond3", "mond4", "track1_rewind", "track2"]:
        for camera_angle in ["center", "right", "left"]:
            
            if dataset == "LEFT" and camera_angle == "center":
                #Do nuthin
                continue
                
            elif dataset == "mond2" and camera_angle == "center":
                #Do nuthin
                use_2 = True
                continue
            
            elif dataset == "track1_rewind" and camera_angle == "center":
                #Do nuthin
                use_2 = False
                use_3 = True
                continue

            elif use_2 != True:
                #model = load_model("curr_best_model.h5")
                np_images_new, np_steering_new = load_dataset_simulator.load_dataset(camera_angle,dataset)

                np_images = np.concatenate((np_images, np_images_new))
                np_steering = np.concatenate((np_steering, np_steering_new))

            elif use_2 and use_3 != True:
                continue
                #model = load_model("curr_best_model.h5")
                np_images_new, np_steering_new = load_dataset_simulator.load_dataset(camera_angle,dataset)

                np_images_2 = np.concatenate((np_images_2, np_images_new))
                np_steering_2 = np.concatenate((np_steering_2, np_steering_new))
                
            elif use_3:
                continue
                #model = load_model("curr_best_model.h5")
                np_images_new, np_steering_new = load_dataset_simulator.load_dataset(camera_angle,dataset)

                np_images_3 = np.concatenate((np_images_3, np_images_new))
                np_steering_3 = np.concatenate((np_steering_3, np_steering_new))



            #print("Currently loading dataset: ", dataset, ", angle: ", camera_angle, ".")
            #np_images, np_steering = load_dataset_simulator.load_dataset(camera_angle,dataset)

            #print("Training the model...")
            #history = model.fit(x=np_images, y=np_steering, epochs=50, batch_size=5, callbacks=[checkpoint, csv_logger], validation_data=(np_val_images, np_val_steering))


            #len_tot_images = len(new_np_images) + len(np_images)
            #len_tot_steering = len(new_np_steering) + len(np_steering)
            #np_images = np.concatenate((np_images, new_np_images))
            #np_steering = np.concatenate((np_steering, new_np_steering))

            #print("Saving the model...")
            #save_load_model.save_model(model, "trained_model_simulator")
            #print("Length of dataset so far: ", len(np_images), " images, ", len(np_steering), " steering angles.")


    print("Saving the model...")

    print("Length of images: ", len(np_images))
    np.save("np_images",np_images)

    print("Length of steer: ", len(np_steering))
    np.save("np_steering",np_steering)
    
    #print("Length of images 2: ", len(np_images_2))
    #np.save("np_images_2",np_images_2)

    #print("Length of steer 2: ", len(np_steering_2))
    #np.save("np_steering_2",np_steering_2)
    
    #print("Length of images 3: ", len(np_images_3))
    #np.save("np_images_3",np_images_3)

    #print("Length of steer 3: ", len(np_steering_3))
    #np.save("np_steering_3",np_steering_3)
    
    #print("Length of images val: ", len(np_val_images))
    #np.save("np_val_images",np_val_images)
    
    #print("Length of steer val: ", len(np_val_steering))
    #np.save("np_val_steering",np_val_steering)
    
    #save_load_model.save_model(model, "trained_model_wednesday")

    #plt.hist(np_steering, bins=100)
    #plt.show()

    #print("Creating heatmap...")
    #heat_map.heat_map()

    #with open('history1.txt', 'wb') as file_pi: #Saved_history/
    #    pickle.dump(history.history, file_pi)

    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
