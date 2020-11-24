from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image as kivy_image
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager,Screen
from kivy.core.window import Window
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_1
from kivy.uix.filechooser import FileChooserIconView
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg as FigureCanvasKivyAgg_1
import sys
from subprocess import call
from kivy.uix.popup import Popup
import csv 
import xml.etree.ElementTree as ET 
from PIL import Image as pil_image
import os
import shutil
from functools import partial
from kivy.logger import Logger
from tensorflow.keras.models import model_from_json




import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator	
Window.clearcolor=(1,1,1,1)

class IntroPage(BoxLayout):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.orientation="vertical"
		self.padding=50
		self.ttk_logo=kivy_image(source="Images/logo.png")
		self.intro_label=Label(text="     GUI Deep Learning Toolkit\n                   Mark-1",color=(0,0,0,1),font_size="30sp")
		self.intro_btn=Button(text="Continue",on_press=self.go_to_second_page,size_hint=(0.3,0.3),pos_hint={"center_x":0.5,"center_y":0.1},font_size=20,background_color=(0,0,0,1))

		self.add_widget(self.ttk_logo)
		self.add_widget(self.intro_label)
		self.add_widget(self.intro_btn)

	def go_to_second_page(self,obj):
		main_app.screen_manager.current="Second"



class elements:
	def __init__(self,class_num):
		self.class_name=TextInput(hint_text="Enter Class {} name:".format(class_num),font_size="20sp",background_color=(0,0,0,1),foreground_color = (1,1,1,1))
		self.dir_inp=TextInput(hint_text="Enter the Directory for Class {}".format(class_num),font_size="20sp",background_color=(0,0,0,1),foreground_color = (1,1,1,1))
		self.split_inp=TextInput(hint_text="Enter train split %",font_size="20sp",background_color=(0,0,0,1),foreground_color = (1,1,1,1))
		self.button=Button(text="Click to Split Dataset",on_press=self.split_data,background_color=(0,0,0,1))
		self.train_dir=""
		self.test_dir=""


	

	def path_check(self,path):
		if os.path.exists(path):
			pass
		else:
			os.makedirs(path)

	def split_data(self,obj):
		if len(self.dir_inp.text)>0 and len(self.split_inp.text)>0:
			self.train_split=int(self.split_inp.text)/100
			self.dir_inp.text=self.dir_inp.text.replace("\\","/")
			self.test_dir=self.dir_inp.text+"/test/"
			self.train_dir=self.dir_inp.text+"/train/"
			self.path_check(self.train_dir)
			self.path_check(self.test_dir)
			print(self.train_dir)
			print(self.test_dir)
			total_img_files=0
			for file in os.listdir(self.dir_inp.text):
				if file.lower().endswith(".png") or file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
					total_img_files+=1
				else:
					pass

			for file in os.listdir(self.dir_inp.text):
				if(file.lower().endswith(".png") or file.lower().endswith(".jpg") or file.lower().endswith(".jpeg")):
					if(len(os.listdir(self.train_dir))<(round(total_img_files*self.train_split))):
						shutil.move(self.dir_inp.text+"/"+file, self.train_dir+file)
					else:
						shutil.move(self.dir_inp.text+"/"+file, self.test_dir+file)
					
						
		else:
			popup = Popup(title='Warning',
			    content=Label(text='Please check Directory or Train Split %'),
			    size_hint=(0.5,0.5))
			popup.open()

	



class SecondPage(GridLayout):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.rows=4
		self.cols=4
		self.padding=50
		self.spacing=10
		# self.row_force_default=True
		# self.row_default_height=50
	
		self.class_1=elements("1")
		self.class_2=elements("2")
		self.class_3=elements("3")

		# self.third_pg_btn=Button

		self.add_widget(self.class_1.class_name)
		self.add_widget(self.class_1.dir_inp)
		self.add_widget(self.class_1.split_inp)
		self.add_widget(self.class_1.button)

		self.add_widget(self.class_2.class_name)
		self.add_widget(self.class_2.dir_inp)
		self.add_widget(self.class_2.split_inp)
		self.add_widget(self.class_2.button)

		self.add_widget(self.class_3.class_name)
		self.add_widget(self.class_3.dir_inp)
		self.add_widget(self.class_3.split_inp)
		self.add_widget(self.class_3.button)

		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Button(text="Continue",on_press=self.go_to_third_page,background_color=(0,0,0,1),font_size="20sp"))

	def go_to_third_page(self,obj):
		main_app.screen_manager.current="Third"

class ThirdPage(GridLayout):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		# self.orientation="vertical"
		self.rows=3	
		self.cols=3
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())

		self.updating_btn=Button(text="Update Attributes",on_press=self.updating,background_color=(0,0,0,1),font_size="20sp")
		self.add_widget(self.updating_btn)
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		
	def updating(self,obj):
		self.clear_widgets()
		self.rows=4
		self.cols=2
		self.padding=50
		self.spacing=10
		self.class_1_train=Button(text="Annotate Class 1 Train Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.open_labelimg, main_app.second_page.class_1.train_dir))
		self.class_2_train=Button(text="Annotate Class 2 Train Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.open_labelimg, main_app.second_page.class_2.train_dir))
		self.class_3_train=Button(text="Annotate Class 3 Train Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.open_labelimg, main_app.second_page.class_3.train_dir))
		self.class_1_test=Button(text="Annotate Class 1 Test Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.open_labelimg, main_app.second_page.class_1.test_dir))
		self.class_2_test=Button(text="Annotate Class 2 Test Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.open_labelimg, main_app.second_page.class_2.test_dir))
		self.class_3_test=Button(text="Annotate Class 3 Test Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.open_labelimg, main_app.second_page.class_3.test_dir))
		self.add_widget(self.class_1_train)
		self.add_widget(self.class_1_test)
		self.add_widget(self.class_2_train)
		self.add_widget(self.class_2_test)
		self.add_widget(self.class_3_train)
		self.add_widget(self.class_3_test)

		self.add_widget(Label())
		self.add_widget(Button(text="Continue",font_size="20sp",on_press=self.go_to_fourth_page,background_color=(0,0,0,1)))

	def go_to_fourth_page(self,obj):
		main_app.screen_manager.current="Fourth"


	def open_labelimg(self,path,obj):

		if len(path)>0:
			call(["python",'LabelImg/labelImg-master/labelImg.py',path])
		else:
			popup = Popup(title='Warning',  
				content=Label(text='Please Enter a Directory'),
				size_hint=(0.3,0.3))
			popup.open()

class FourthPage(GridLayout):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.rows=3	
		self.cols=3
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())

		self.updating_btn=Button(text="Update Attributes",on_press=self.updating,background_color=(0,0,0,1),font_size="20sp")
		self.add_widget(self.updating_btn)
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())

	def updating(self,obj):
		self.clear_widgets()
		self.rows=4
		self.cols=2
		self.padding=50
		self.spacing=10
		self.class_1_train=Button(text="Crop Annotated Class 1 Train Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.crop_annotaed_imgs, main_app.second_page.class_1.train_dir,"train",main_app.second_page.class_1.class_name))
		self.class_2_train=Button(text="Crop Annotated Class 2 Train Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.crop_annotaed_imgs, main_app.second_page.class_2.train_dir,"train",main_app.second_page.class_2.class_name))
		self.class_3_train=Button(text="Crop Annotated Class 3 Train Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.crop_annotaed_imgs, main_app.second_page.class_3.train_dir,"train",main_app.second_page.class_3.class_name))
		self.class_1_test=Button(text="Crop Annotated Class 1 Test Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.crop_annotaed_imgs, main_app.second_page.class_1.test_dir,"test",main_app.second_page.class_1.class_name))
		self.class_2_test=Button(text="Crop Annotated Class 2 Test Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.crop_annotaed_imgs, main_app.second_page.class_2.test_dir,"test",main_app.second_page.class_2.class_name))
		self.class_3_test=Button(text="Crop Annotated Class 3 Test Images",background_color=(0,0,0,1),font_size="20sp",on_press=partial(self.crop_annotaed_imgs, main_app.second_page.class_3.test_dir,"test",main_app.second_page.class_3.class_name))
		self.add_widget(self.class_1_train)
		self.add_widget(self.class_1_test)
		self.add_widget(self.class_2_train)
		self.add_widget(self.class_2_test)
		self.add_widget(self.class_3_train)
		self.add_widget(self.class_3_test)
		self.dataset_folder_name="Mark1_dataset/"

		self.add_widget(Label())
		self.add_widget(Button(text="Continue",on_press=self.go_to_fifth_page,background_color=(0,0,0,1)))

	def go_to_fifth_page(self,obj):
		main_app.screen_manager.current="Fifth"

	def path_check(self,path):
		if os.path.exists(path):
			pass
		else:
			os.makedirs(path)

	def crop_annotaed_imgs(self,path,trrte,c_name,obj):
		print("a",trrte)
		for file in os.listdir(path):
			if file.lower().endswith(".xml"):
				tree = ET.parse(path+"/"+file)
				root = tree.getroot()

				for ind,bndbo in enumerate(root.findall('object/bndbox')):
					li=[]
					for axis in bndbo:
						li.append(int(axis.text))
					im_name=root.find("filename")
					if trrte=="train":
						self.new_path=self.dataset_folder_name+"train/"+c_name.text+"/"
						print(self.new_path)
						self.path_check(self.new_path)
						im = pil_image.open(path+"/"+im_name.text)
						region = im.crop(li)

						region.save(self.new_path+im_name.text.split(".")[0]+"_piece{}.png".format(str(ind)))
					elif trrte=="test":
						self.new_path=self.dataset_folder_name+"test/"+c_name.text+"/"
						print(self.new_path)
						self.path_check(self.new_path)
						im = pil_image.open(path+"/"+im_name.text)
						region = im.crop(li)

						region.save(self.new_path+im_name.text.split(".")[0]+"_piece{}.png".format(str(ind)))


class FifthPage(GridLayout):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.rows=3	
		self.cols=3
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.updating_btn=Button(text="Update Attributes",on_press=self.updating,background_color=(0,0,0,1),font_size="20sp")
		self.add_widget(self.updating_btn)
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
	def updating(self,obj):
		self.clear_widgets()
		self.rows=6
		self.cols=2
		self.padding=50
		self.spacing=10
		self.add_widget(Label(text="Num_of_Epochs: ",color=(0,0,0,1),font_size="20sp"))
		self.epoch_inp=TextInput(hint_text="Enter total epochs for training(int)",font_size="20sp",background_color=(0,0,0,1),foreground_color = (1,1,1,1))
		self.add_widget(self.epoch_inp)
		self.add_widget(Label(text="Batch_Size: ",color=(0,0,0,1),font_size="20sp"))
		self.batch_size_inp=TextInput(hint_text="Enter batch size for training(int)",font_size="20sp",background_color=(0,0,0,1),foreground_color = (1,1,1,1))
		self.add_widget(self.batch_size_inp)
		self.add_widget(Label(text="Learning_Rate: ",color=(0,0,0,1),font_size="20sp"))
		self.learning_rate_inp=TextInput(hint_text="Enter learning rate for training(float)",font_size="20sp",background_color=(0,0,0,1),foreground_color = (1,1,1,1))
		self.add_widget(self.learning_rate_inp)
		self.add_widget(Label(text="Drop-out_Rate: ",color=(0,0,0,1),font_size="20sp"))

		self.drop_out_rate_inp=TextInput(hint_text="Enter Drop-out rate for training(%)",font_size="20sp",background_color=(0,0,0,1),foreground_color = (1,1,1,1))
		self.add_widget(self.drop_out_rate_inp)
		self.add_widget(Label(text="Choose_Model: ",color=(0,0,0,1),font_size="20sp"))
		self.choosen_model=Label(text="InceptionV3",color=(0,0,0,1),font_size="20sp")
		self.add_widget(self.choosen_model)
		
		self.add_widget(Label())

		self.add_widget(Button(text="Continue",font_size="20sp",on_press=self.go_to_sixth_page,background_color=(0,0,0,1)))

	def go_to_sixth_page(self,obj):
		if(len(self.epoch_inp.text)>0) and (len(self.batch_size_inp.text)>0) and (len(self.learning_rate_inp.text)>0):
			main_app.screen_manager.current="Sixth"
		else:
			popup = Popup(title='Warning',  
				content=Label(text='Please Enter Num_of_Epochs or Batch_Size or Learning_Rate'),
				size_hint=(0.6,0.3))
			popup.open()


class SixthPage(BoxLayout):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)

		self.orientation="vertical"
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.updating_btn=Button(text="Start Training",on_press=self.updating,pos_hint={"center_x":0.5,"center_y":0.5},size_hint=(0.5,0.5),background_color=(0,0,0,1),font_size="20sp")
		self.add_widget(self.updating_btn)
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
		self.add_widget(Label())
	def updating(self,obj):
		# print(main_app.fourth_page.dataset_folder_name)
		# print(int(main_app.fifth_page.epoch_inp.text))
		# print(int(main_app.fifth_page.batch_size_inp.text))
		# print(float(main_app.fifth_page.learning_rate_inp.text))
		# print(float(int(main_app.fifth_page.drop_out_rate_inp.text)/100))
		self.clear_widgets()

		self.orientation="vertical"
		local_weights_file="model_weights/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

		pre_trained_model=InceptionV3(input_shape=(150,150,3),
		                             include_top=False,
		                             weights=None)

		pre_trained_model.load_weights(local_weights_file)

		for layer in pre_trained_model.layers:
		    layer.trainable=False
		    
		last_layer=pre_trained_model.get_layer("mixed7")

		last_output=last_layer.output

		x=layers.Flatten()(last_output)
		x=layers.Dense(1024,activation="relu")(x)
		x = layers.Dropout(float(int(main_app.fifth_page.drop_out_rate_inp.text)/100))(x)                  
		x=layers.Dense(1,activation="sigmoid")(x)

		self.model=Model(pre_trained_model.input,x)

		self.model.compile(optimizer=RMSprop(lr=float(main_app.fifth_page.learning_rate_inp.text)),
		             loss="categorical_crossentropy",
		             metrics=["acc"])

		from tensorflow.keras.preprocessing.image import ImageDataGenerator

		train_datagen=ImageDataGenerator(
		      rescale=1./255,
		      rotation_range=40,
		      width_shift_range=0.2,
		      height_shift_range=0.2,
		      shear_range=0.2,
		      zoom_range=0.2,
		      horizontal_flip=True)

		train_generator=train_datagen.flow_from_directory(
		                main_app.fourth_page.dataset_folder_name+"train/",target_size=(150,150),class_mode="categorical")

		validation_datagen=ImageDataGenerator(rescale=1/255)

		self.validation_generator=validation_datagen.flow_from_directory(
		                main_app.fourth_page.dataset_folder_name+"test/",target_size=(150,150),class_mode="categorical")
		history=self.model.fit_generator(train_generator,validation_data=self.validation_generator,
		                           epochs=int(main_app.fifth_page.epoch_inp.text),verbose=2)

		scores = self.model.evaluate_generator(self.validation_generator)
		print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

		acc=history.history["acc"]
		val_acc=history.history["val_acc"]

		loss=history.history["loss"]
		val_loss=history.history["val_loss"]
		epochs=range(len(acc))
		plt.subplot(211)
		plt.title("Training History")
		plt.plot(epochs,acc,"-b", label="training_acc")
		plt.plot(epochs,val_acc, "-r", label="validation_acc")
		plt.legend(loc="upper left")
		
		plt.subplot(212)
		# self.add_widget(FigureCanvasKivyAgg(plt.gcf()))
		plt.plot(epochs,loss,"-b", label="training_loss")
		plt.plot(epochs,val_loss, "-r", label="validation_loss")
		plt.legend(loc="upper left")
		
		
		self.add_widget(FigureCanvasKivyAgg(plt.gcf()))
		
		self.model_save_btn=Button(text="Save model",background_color=(0,0,0,1),font_size="20sp",on_press=self.save_model,size_hint=(0.2,0.2),pos_hint={"center_x":0.5,"center_y":0.8})
		self.add_widget(self.model_save_btn)
		self.model_load_btn=Button(text="Load model",background_color=(0,0,0,1),font_size="20sp",on_press=self.load_model,size_hint=(0.2,0.2),pos_hint={"center_x":0.5,"center_y":0.8})
		self.add_widget(self.model_load_btn)
		self.add_widget(Button(text="Quit",background_color=(0,0,0,1),font_size="20sp",on_press=self.quit_app,size_hint=(0.2,0.2),pos_hint={"center_x":0.9,"center_y":1.0}))
	def path_check(self,path):
		if os.path.exists(path):
			pass
		else:
			os.makedirs(path)
	def quit_app(self,obj):
	    App.get_running_app().stop()	
	def save_model(self,obj):
		self.model_json = self.model.to_json()
		self.path_check("saved_models/")
		with open("saved_models/model.json", "w") as json_file:
		    json_file.write(self.model_json)
		
		self.model.save_weights("saved_models/model.h5")
		print("Saved model to disk")
	def load_model(self,obj):
		json_file = open('saved_models/model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("saved_models/model.h5")
		print("Loaded model from disk")

		# evaluate loaded model on test data
		loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
		score = loaded_model.evaluate_generator(self.validation_generator)
		print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))








	

class MainApp(App):
	def build(self):
		self.screen_manager=ScreenManager()
		self.intro_page=IntroPage()
		screen=Screen(name="Intro")
		screen.add_widget(self.intro_page)
		self.screen_manager.add_widget(screen)

		self.second_page=SecondPage()
		screen=Screen(name="Second")
		screen.add_widget(self.second_page)
		self.screen_manager.add_widget(screen)

		self.third_page=ThirdPage()
		screen=Screen(name="Third")
		screen.add_widget(self.third_page)
		self.screen_manager.add_widget(screen)

		self.fourth_page=FourthPage()
		screen=Screen(name="Fourth")
		screen.add_widget(self.fourth_page)
		self.screen_manager.add_widget(screen)

		self.fifth_page=FifthPage()
		screen=Screen(name="Fifth")
		screen.add_widget(self.fifth_page)
		self.screen_manager.add_widget(screen)

		self.sixth_page=SixthPage()
		screen=Screen(name="Sixth")
		screen.add_widget(self.sixth_page)
		self.screen_manager.add_widget(screen)

		

		return self.screen_manager


if __name__=="__main__":
	main_app=MainApp()
	main_app.run()