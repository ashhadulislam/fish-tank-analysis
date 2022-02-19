import streamlit as st
from lib import commons
import torch
import os
import FCN_NetModel as FCN # The net Class
import CategoryDictionary as CatDic
import cv2
import numpy as np
from PIL import Image

def get_top_bottom(matrix):
    '''
    return the 4 corners where matrix values is 1
    '''
    top=-1
    bottom=-1
    count=0
    for row in matrix:        
        if any(row)==1:
            top=count
            break
        count+=1
    count=0
    for row in np.flip(matrix, 0):
        if any(row)==1:
            bottom=count
            break
        count+=1
    bottom=matrix.shape[0]-bottom
    return top,bottom
        
            

def app():
    header=st.container()
    result_all = st.container()


    OutDir="Out/" # Folder of output

    UseGPU=False # Use GPU or CPU  for prediction (GPU faster but demend nvidia GPU and CUDA installed else set UseGPU to False)
    FreezeBatchNormStatistics=False # wether to freeze the batch statics on prediction  setting this true or false might change the prediction mostly False work better
    OutEnding="" # Add This to file name
    if not os.path.exists(OutDir): os.makedirs(OutDir) # Create folder for trained weight
    #-----------------------------------------Location of the pretrain model-----------------------------------------------------------------------------------
    Trained_model_path ="models/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"

    ##################################Load net###########################################################################################
    #---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
    Net=FCN.Net(CatDic.CatNum) # Create net and load pretrained encoder path
    if UseGPU==True:
        print("USING GPU")
        Net.load_state_dict(torch.load(Trained_model_path))
    else:
        print("USING CPU")
        Net.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))
    #--------------------------------------------------------------------------------------------------------------------------






    with header:
        st.subheader("Test Water Tank Image")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        if image_file is not None:
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                          "filesize":image_file.size}
            st.write(file_details)

            # To View Uploaded Image
            st.image(commons.load_image(image_file)
                ,width=250
                )
            print("Image file is it showing location?",image_file)
            name=image_file.name


            # Im=cv2.imread(image_file)
            Im = Image.open(image_file)
            # Im=cv2.imread(image_file.name)
            Im = np.asarray(Im)
            
            h,w,d=Im.shape
            print("Read iamge file shape",h,w,d)
            r=np.max([h,w])
            if r>840: # Image larger then 840X840 are shrinked (this is not essential, but the net results might degrade when using to large images
                fr=840/r
                Im=cv2.resize(Im,(int(w*fr),int(h*fr)))
            Imgs=np.expand_dims(Im,axis=0)
            if not (type(Im) is np.ndarray): 
                print("Not an image")            
        # #................................Make Prediction......................................................................................
            with torch.autograd.no_grad():
                  OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNormStatistics) # Run net inference and get prediction
        # #.......................Save prediction on fil
            print("Saving output to: " + OutDir)
            accepted_names=["Liquid GENERAL","Liquid Suspension","Powder","Solid GENERAL"]
            for nm in OutLbDict:
                
                if nm not in accepted_names:
                    continue
                    
                Lb=OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8)

                if Lb.mean()<0.001: continue
                if nm=='Ignore': continue
        #         if nm!="Filled": continue
                # Lb is a 2d matrix which is made 1 where the 
                # material is present

                ImOverlay1 = Im.copy()
                ImOverlay1[:, :, 0][Lb==1] = 255
                ImOverlay1[:, :, 1][Lb==1] = 0
                ImOverlay1[:, :, 2][Lb==1] = 255
                
                FinIm=np.concatenate([Im,ImOverlay1],axis=1)

                st.image(ImOverlay1, caption=nm,width=250)




                OutPath = OutDir + "//" + nm+"/"

                if not os.path.exists(OutPath): os.makedirs(OutPath)
                OutName=OutPath+name[:-4]+OutEnding+".png"
                cv2.imwrite(OutName,FinIm)
                
                # if nm=="Filled":
                #     print("nm is ",nm)
                #     t,b=get_top_bottom(Lb)
                #     print("top:",t,"bottom",b)
                #     print("Height in pixels:",b-t)

        #             print(Im.shape,ImOverlay1.shape)
        #             print(FinIm.shape)
                    # print("******************")
                






            image_for_model = commons.image_loader(image_file)
            print("Loaded image for model")
        else:
            proxy_img_file="data/joplin-tornado_00000001_post_disaster.png"
            st.image(commons.load_image(proxy_img_file),width=250)
            image_for_model=commons.image_loader(proxy_img_file)
            print("Loaded proxy image for model")

    with result_all:    

        model_name="squeezenet"
        num_classes = 2        
        feature_extract = False
        # Initialize the model for this run
        model_ft, input_size = commons.initialize_model(model_name, num_classes,
        					 feature_extract, use_pretrained=True)        
        model_state_path="models/squeezenet_10_pre_vs_post_all.pt"
        
        