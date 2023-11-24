from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (QFileDialog, QWidget, QLabel, QMainWindow, QApplication, QScrollArea, QTableWidget, QTableWidgetItem,QVBoxLayout,)
from PyQt6.QtGui import QPixmap, QImage
import sys
from pathlib import Path
import os
from liblinear_model import linear_predict_proba
from liblinear.liblinearutil import load_model
from tkinter import messagebox
import fitz  # pip install PyMuPDF



class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__() 

        os.system('cls')
        #Container for Real and Gan Images
        self.images = []

        # initialize value of model and clf
        self.loaded_model = None
        self.clf = None

        # result and probability estimates
        self.result = []
        self.prob = []

        # true labels
        self.true_labels = [] 

        QMainWindow().__init__(self)
        self.ui = MainWindow
        self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 646)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        #Get Started Button
        self.getStarted = QtWidgets.QPushButton(parent=self.centralwidget)
        self.getStarted.setGeometry(QtCore.QRect(440, 330, 121, 51))
        self.getStarted.setStyleSheet("* {\n""background: transparent;\n""}")
        self.getStarted.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resources/Button1.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.getStarted.setIcon(icon)
        self.getStarted.setIconSize(QtCore.QSize(110, 110))
        self.getStarted.setObjectName("getStarted")

        #Welcome Page Icons & Labels
        self.icon = QtWidgets.QLabel(parent=self.centralwidget)
        self.icon.setGeometry(QtCore.QRect(370, 340, 261, 261))
        self.icon.setText("")
        self.icon.setPixmap(QtGui.QPixmap("resources/Icon.png"))
        self.icon.setScaledContents(True)
        self.icon.setObjectName("icon")
        self.gallery1 = QtWidgets.QLabel(parent=self.centralwidget)
        self.gallery1.setGeometry(QtCore.QRect(660, 510, 171, 121))
        self.gallery1.setText("")
        self.gallery1.setPixmap(QtGui.QPixmap("resources/gallery1.png"))
        self.gallery1.setScaledContents(True)
        self.gallery1.setObjectName("gallery1")
        self.eye1 = QtWidgets.QLabel(parent=self.centralwidget)
        self.eye1.setGeometry(QtCore.QRect(670, 370, 171, 161))
        self.eye1.setText("")
        self.eye1.setPixmap(QtGui.QPixmap("resources/eye.png"))
        self.eye1.setScaledContents(True)
        self.eye1.setObjectName("eye1")
        self.gallery2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.gallery2.setGeometry(QtCore.QRect(180, 370, 171, 171))
        self.gallery2.setText("")
        self.gallery2.setPixmap(QtGui.QPixmap("resources/gallery.png"))
        self.gallery2.setScaledContents(True)
        self.gallery2.setObjectName("gallery2")
        self.eye2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.eye2.setGeometry(QtCore.QRect(170, 510, 171, 121))
        self.eye2.setText("")
        self.eye2.setPixmap(QtGui.QPixmap("resources/eye1.png"))
        self.eye2.setScaledContents(True)
        self.eye2.setObjectName("eye2")
        self.title = QtWidgets.QLabel(parent=self.centralwidget)
        self.title.setGeometry(QtCore.QRect(210, 120, 601, 51))
        self.title.setStyleSheet("* {\n""font: 700 45pt \"Arial\";\n""}")
        self.title.setText("")
        self.title.setPixmap(QtGui.QPixmap("resources/Welcome to App Name.png"))
        self.title.setScaledContents(True)
        self.title.setObjectName("title")
        self.logoName = QtWidgets.QLabel(parent=self.centralwidget)
        self.logoName.setGeometry(QtCore.QRect(50, 30, 111, 16))
        self.logoName.setStyleSheet("*{\n""font: 700 10pt \"Arial\";\n""color:rgb(255, 255, 255);\n""}")
        self.logoName.setObjectName("logoName")
        self.logo = QtWidgets.QLabel(parent=self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(10, 20, 31, 31))
        self.logo.setText("")
        self.logo.setPixmap(QtGui.QPixmap("resources/openAI-logo.png"))
        self.logo.setScaledContents(True)
        self.logo.setObjectName("logo")
        self.socmedIcons = QtWidgets.QLabel(parent=self.centralwidget)
        self.socmedIcons.setGeometry(QtCore.QRect(870, 30, 101, 16))
        self.socmedIcons.setText("")
        self.socmedIcons.setPixmap(QtGui.QPixmap("resources/Group 2.png"))
        self.socmedIcons.setScaledContents(True)
        self.socmedIcons.setObjectName("socmedIcons")

        #Navbar
        self.homeBar = QtWidgets.QPushButton(parent=self.centralwidget)
        self.homeBar.setGeometry(QtCore.QRect(240, 30, 75, 24))
        self.homeBar.setStyleSheet("* {\n""background: transparent;\n""color: rgb(85, 255, 255)\n""}")
        self.homeBar.setObjectName("homeBar")
        self.datasetBar = QtWidgets.QPushButton(parent=self.centralwidget)
        self.datasetBar.setGeometry(QtCore.QRect(340, 30, 75, 24))
        self.datasetBar.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")
        self.datasetBar.setObjectName("datasetBar")
        self.modelBar = QtWidgets.QPushButton(parent=self.centralwidget)
        self.modelBar.setGeometry(QtCore.QRect(450, 30, 75, 24))
        self.modelBar.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")
        self.modelBar.setObjectName("modelBar")
        self.docuBar = QtWidgets.QPushButton(parent=self.centralwidget)
        self.docuBar.setGeometry(QtCore.QRect(560, 30, 91, 24))
        self.docuBar.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")
        self.docuBar.setObjectName("docuBar")
        self.aboutBar = QtWidgets.QPushButton(parent=self.centralwidget)
        self.aboutBar.setGeometry(QtCore.QRect(690, 30, 51, 24))
        self.aboutBar.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")
        self.aboutBar.setObjectName("aboutBar")

        self.imageLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(70, 90, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.imageLabel.setFont(font)
        self.imageLabel.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")
        self.imageLabel.setObjectName("imageLabel")
        self.resultLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.resultLabel.setGeometry(QtCore.QRect(550, 100, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.resultLabel.setFont(font)
        self.resultLabel.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")
        self.resultLabel.setObjectName("resultLabel")
        self.aibutton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.aibutton.setGeometry(QtCore.QRect(270, 540, 121, 51))
        self.aibutton.setStyleSheet("* {\n""background: transparent\n""}")
        self.aibutton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resources/Ai.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.aibutton.setIcon(icon)
        self.aibutton.setIconSize(QtCore.QSize(105, 105))
        self.aibutton.setObjectName("aibutton")
        self.clearbutton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.clearbutton.setGeometry(QtCore.QRect(130, 540, 121, 51))
        self.clearbutton.setStyleSheet("* {\n""background: transparent\n""}")
        self.clearbutton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("resources/Clear.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.clearbutton.setIcon(icon1)
        self.clearbutton.setIconSize(QtCore.QSize(105, 105))
        self.clearbutton.setObjectName("clearbutton")
        self.Rectangle1 = QtWidgets.QLabel(parent=self.centralwidget)
        self.Rectangle1.setGeometry(QtCore.QRect(60, 90, 401, 521))
        self.Rectangle1.setText("")
        self.Rectangle1.setPixmap(QtGui.QPixmap("resources/Rectangle.png"))
        self.Rectangle1.setScaledContents(True)
        self.Rectangle1.setObjectName("Rectangle1")
        self.Rectangle1.lower()
        self.Rectangle2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.Rectangle2.setGeometry(QtCore.QRect(540, 90, 401, 521))
        self.Rectangle2.setText("")
        self.Rectangle2.setPixmap(QtGui.QPixmap("resources/Rectangle.png"))
        self.Rectangle2.setScaledContents(True)
        self.Rectangle2.setObjectName("Rectangle2")
        self.uploadBar = QtWidgets.QPushButton(parent=self.centralwidget)
        self.uploadBar.setGeometry(QtCore.QRect(10, 95, 511, 511))
        self.uploadBar.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")
        self.uploadBar.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("resources/Wrapper1.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.uploadBar.setIcon(icon2)
        self.uploadBar.setIconSize(QtCore.QSize(580, 490))
        self.uploadBar.setObjectName("uploadBar")
        self.Wrapper = QtWidgets.QLabel(parent=self.centralwidget)
        self.Wrapper.setGeometry(QtCore.QRect(520, 102, 451, 561))
        self.Wrapper.setText("")
        self.Wrapper.setPixmap(QtGui.QPixmap("resources/Wrapper1.png"))
        self.Wrapper.setScaledContents(True)
        self.Wrapper.setObjectName("Wrapper")
        self.uploadButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.uploadButton.setGeometry(QtCore.QRect(200, 240, 131, 121))
        self.uploadButton.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")
        self.uploadButton.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("resources/gallery2.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.uploadButton.setIcon(icon3)
        self.uploadButton.setIconSize(QtCore.QSize(110, 110))
        self.uploadButton.setObjectName("uploadButton")
        self.eye3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.eye3.setGeometry(QtCore.QRect(650, 300, 191, 211))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.eye3.setFont(font)
        self.eye3.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")
        self.eye3.setText("")
        self.eye3.setPixmap(QtGui.QPixmap("resources/eye3.png"))
        self.eye3.setScaledContents(True)
        self.eye3.setObjectName("eye3")

        self.eye4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.eye4.setGeometry(QtCore.QRect(690, 300, 191, 211))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.eye4.setFont(font)
        self.eye4.setText("")
        self.eye4.setObjectName("eye4")
        self.eye3.setStyleSheet("* {\ncolor: rgb(255, 255, 255)\n""}")


        self.aboutBar_2 = QtWidgets.QPushButton(parent=self.centralwidget)
        self.aboutBar_2.setGeometry(QtCore.QRect(130, 340, 261, 111))
        self.aboutBar_2.setStyleSheet("* {\n""background: transparent;\n""color: rgb(169,169,169)\n""}")
        self.aboutBar_2.setObjectName("aboutBar_2")
        self.aboutBar_2.setText("Click to upload your Image")
        self.uploadLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.uploadLabel.setGeometry(QtCore.QRect(110, 575, 321, 41))
        self.uploadLabel.setText("By uploading an image,  you agree to our Terms of Service")
        font = QtGui.QFont()
        font.setPointSize(9)
        self.uploadLabel.setFont(font)
        self.uploadLabel.setStyleSheet("* {\n""background: transparent;\n""color: rgb(255, 255, 255)\n""}")

        # notifications
        self.notif = QtWidgets.QLabel(parent=self.centralwidget)
        self.notif.setGeometry(QtCore.QRect(855, 555, 150, 45))
        self.notif.setText("")
        self.notif.setPixmap(QtGui.QPixmap("resources/loading.png"))
        self.notif.setScaledContents(True)
        self.notif.setObjectName("eye2")

        self.setWelcomePage()

        # select model
        self.modelBar.clicked.connect(self.select_model)
        self.docuBar.clicked.connect(self.setDocumentationPage)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "illuscan"))
        self.logoName.setText(_translate("MainWindow", "illuscan"))
        self.homeBar.setText(_translate("MainWindow", "Home"))
        self.datasetBar.setText(_translate("MainWindow", "Dataset"))
        self.modelBar.setText(_translate("MainWindow", "Model"))
        self.docuBar.setText(_translate("MainWindow", "Documentation"))
        self.aboutBar.setText(_translate("MainWindow", "About"))
    
    def setWelcomePage(self):
        MainWindow.setStyleSheet("#centralwidget {\n""background-image: url(resources/Home page.png);\n""}")
        self.icon.hide()
        self.getStarted.show()
        self.title.hide()
        self.gallery1.hide()
        self.gallery2.hide()
        self.eye1.hide()
        self.eye2.hide()
        self.aboutBar.clicked.connect(self.setAboutPage)
        self.imageLabel.hide()
        self.resultLabel.hide()
        self.aibutton.hide()
        self.clearbutton.hide()
        self.Rectangle1.hide()
        self.Rectangle2.hide()
        self.uploadBar.hide()
        self.Wrapper.hide()
        self.uploadButton.hide()
        self.eye3.hide()
        self.uploadLabel.hide()
        self.aboutBar_2.hide()
        self.eye4.hide()
        self.notif.hide()

        scroll_area = self.centralwidget.findChild(QScrollArea)
        if scroll_area and scroll_area.isVisible():
            scroll_area.hide()


        
        self.getStarted.clicked.connect(self.setHomePage)
        #MainWindow.setStyleSheet("#centralwidget {\n""background-image: url(resources/Get Started.jpg);\n""}")
        self.aboutBar.clicked.connect(self.setAboutPage)

    def setAboutPage(self):
        MainWindow.setStyleSheet("#centralwidget {\n""background-image: url(resources/Get Started.jpg);\n""}")
        self.icon.show()
        self.getStarted.hide()
        self.title.show()
        self.gallery1.show()
        self.gallery2.show()
        self.eye1.show()
        self.eye2.show()
        self.imageLabel.hide()
        self.resultLabel.hide()
        self.aibutton.hide()
        self.clearbutton.hide()
        self.Rectangle1.hide()
        self.Rectangle2.hide()
        self.uploadBar.hide()
        self.Wrapper.hide()
        self.uploadButton.hide()
        self.eye3.hide()
        self.aboutBar_2.hide()
        self.uploadLabel.hide()
        self.notif.hide()
        self.homeBar.clicked.connect(self.setWelcomePage)

    def setHomePage(self):
        MainWindow.setStyleSheet("#centralwidget {\n""background-image: url(resources/Get Started.jpg);\n""}")
        self.icon.hide()
        self.getStarted.hide()
        self.icon.hide()
        self.getStarted.hide()
        self.title.hide()
        self.gallery1.hide()
        self.gallery2.hide()
        self.eye1.hide()
        self.eye2.hide()
        self.notif.hide()
        self.imageLabel.show()
        self.resultLabel.show()
        self.aibutton.raise_()
        self.aibutton.show()
        self.aibutton.clicked.connect(self.predict_result)
        self.clearbutton.show()
        self.clearbutton.raise_()
        self.clearbutton.clicked.connect(self.clear_image)
        self.Rectangle1.show()
        self.Rectangle2.show()
        self.uploadBar.show()
        self.Wrapper.show()
        self.uploadButton.show()
        self.eye3.show()
        self.uploadLabel.show()
        self.aboutBar_2.show()
        self.homeBar.clicked.connect(self.setWelcomePage)
        self.uploadButton.clicked.connect(self.uploadImage)

        self.image_container = QScrollArea(self.centralwidget)
        self.image_container.setStyleSheet("background-color: transparent")
        self.image_container.setGeometry(82, 110, 360, 420)  # Adjust the size
        self.image_container.setWidgetResizable(True)

        self.image_grid_layout = QtWidgets.QGridLayout()
        self.image_grid_layout.setColumnMinimumWidth(0, 0)  # Set the minimum width to 0 for the first column
        self.image_grid_layout.setColumnMinimumWidth(1, 100)
        self.image_grid_layout.setColumnMinimumWidth(2, 100)

        container_widget = QtWidgets.QWidget()
        container_widget.setStyleSheet("border: none")
        container_widget.setLayout(self.image_grid_layout)
        self.image_container.setWidget(container_widget)

        self.loadingDetection = QtWidgets.QLabel(parent=self.Wrapper)
        self.loadingDetection.setGeometry(QtCore.QRect(160, 270, 261, 111))
        self.loadingDetection.setStyleSheet("* {\n""background: transparent;\n""color: rgb(169,169,169);\n"" font-size: 16px;}")
        self.loadingDetection.setObjectName("loading")
        self.loadingDetection.setText("Detecting Image")

    
    
    
    def setDocumentationPage(self):
        # Customize UI elements for the documentation page
        MainWindow.setStyleSheet("#centralwidget {\n""background-image: url(resources/Get Started.jpg);\n""}")
        self.getStarted.hide()
        self.imageLabel.hide()
        self.resultLabel.hide()
        self.aibutton.hide()
        self.clearbutton.hide()
        self.Rectangle1.hide()
        self.Rectangle2.hide()
        self.uploadBar.hide()
        self.Wrapper.hide()
        self.uploadButton.hide()
        self.eye3.hide()
        self.aboutBar_2.hide()
        self.uploadLabel.hide()
        self.notif.hide()
        self.homeBar.clicked.connect(self.setWelcomePage)
        
        self.displayDocumentation()

    def displayDocumentation(self):
        doc = fitz.open('documentation/documentation.pdf')

        scroll_area = QScrollArea(self.centralwidget)
        scroll_area.setGeometry(150, 100, 700, 520)   #Adjust the size x, y, width, height
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)

        for i in range(len(doc)):
            page = doc.load_page(i)
            pixmap = page.get_pixmap()

            label = QLabel()
            label.setPixmap(QPixmap.fromImage(
                QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride,
                       QImage.Format.Format_RGB888 if pixmap.alpha == 0 else QImage.Format.Format_RGBA8888))
            )
           
            layout.addWidget(label)

        scroll_area.show()

    # upload images and display them in the left panel of home page
    def uploadImage(self):
        home_dir = str(Path.home())
        fname, _ = QFileDialog.getOpenFileNames(self, 'Open file', home_dir)

        if fname:
            self.image_container.show()
            self.uploadButton.hide()
            self.aboutBar_2.hide()

            for idx, file in enumerate(fname):
                pixmap = QPixmap(file)
                pixmap = pixmap.scaled(110, 110)

                label = QtWidgets.QLabel()
                label.setPixmap(pixmap)

                self.images.append(file)

                row = idx // 3
                col = idx % 3

                if col == 0:
                    self.image_grid_layout.setColumnMinimumWidth(0, 0)

                self.image_grid_layout.addWidget(label, row, col)

            if (len(fname) - 1) % 3 != 0:
                row += 1
                for col in range((len(fname) - 1) % 3, 3):
                    self.image_grid_layout.addWidget(QtWidgets.QLabel(), row, col)

                self.image_grid_layout.setContentsMargins(0, 2, 0, 2)  

    # get the basename (example: /user/hi.png => hi.png)
    def get_basename(self, images):
        images_basename = [os.path.basename(images) for images in self.images]

        return images_basename
 

    # detect whether an image is gan or real with probability estimates    
    def predict_result(self): 
        try: 
            if len(self.images) != 0: 
                if self.loaded_model is None: 
                    print("No model loaded") 
                else:
                    self.loadingDetection.show()  # Show the label before processing starts
                    QApplication.processEvents()# Show the label before processing starts
                    result, likelihood = linear_predict_proba(self.images, self.loaded_model, self.clf)

                    for prob, pred in zip(likelihood, result): 
                        self.prob.append(prob)
                        self.result.append(pred)
                    QApplication.processEvents() 
                self.display_result() 
        except Exception as e:
            print(f"Something went wrong! : {e}")
        finally:
            self.loadingDetection.hide()
    # reset
    def clear_image(self):
        if len(self.images) != 0:
            self.images.clear()
            self.prob.clear()
            self.result.clear()
            self.eye3.show()
            self.loadingDetection.hide()

            try:
                self.result_table.setRowCount(0)
                self.result_table.clearContents()

                # Clear the image labels from the layout
                for i in reversed(range(self.image_grid_layout.count())):
                    self.image_grid_layout.itemAt(i).widget().setParent(None)   

                self.result_table.hide()
                self.processed_images_label.hide()
                self.detected_real_label.hide()
                self.detected_gan_label.hide()
            except Exception as e:
                print(f"Something went wrong! : {e}") 
           
            self.uploadButton.show()
            self.image_container.hide()
            self.aboutBar_2.show()
            os.system('cls')
            messagebox.showinfo(message=f"Images {len(self.images)}")     
        else:
            messagebox.showinfo(message="Images already cleared")

    # display the result in table
    def display_result(self):
        
        # Create labels for statistics
        self.processed_images_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.processed_images_label.setGeometry(670, 440, 200, 20)
        self.detected_real_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.detected_real_label.setGeometry(670, 455, 200, 20)
        self.detected_gan_label = QtWidgets.QLabel(parent=self.centralwidget)
        self.detected_gan_label.setGeometry(670, 470, 200, 20)

        self.processed_images_label.setStyleSheet("color:white; font-size: 11pt")
        self.detected_real_label.setStyleSheet("color:white; font-size: 11pt")
        self.detected_gan_label.setStyleSheet("color:white; font-size: 11pt")  

        if len(self.result) > 0:
            self.result_table = QTableWidget(self.centralwidget)
            self.result_table.setRowCount(len(self.result))
            self.result_table.setColumnCount(4)
            self.result_table.setHorizontalHeaderLabels(["Image Name", "Real Probability", "GAN Probability", "Prediction"])
            self.result_table.setStyleSheet("background-color: transparent")
            self.result_table.setGeometry(560, 115, 363, 320)  # Adjust the size x, y, width, height
            self.result_table.horizontalHeader().setStyleSheet("background-color: transparent")
            self.result_table.verticalHeader().setStyleSheet("background-color: transparent")

            for row, (image_name, prediction, probability) in enumerate(zip(self.get_basename(self.images), self.result, self.prob)):
                self.result_table.setItem(row, 0, QTableWidgetItem(image_name))
                self.result_table.setItem(row, 1, QTableWidgetItem(f"{(probability[1] * 100):.2f}"))
                self.result_table.setItem(row, 2, QTableWidgetItem(f"{(probability[0] * 100):.2f}"))
                self.result_table.setItem(row, 3, QTableWidgetItem(prediction))

            self.eye3.hide()
            self.result_table.resizeColumnsToContents()
            self.result_table.show()

            # Update labels for statistics
            self.processed_images_text = f"Processed Images: {len(self.images)}"
            self.detected_real_text = f"Detected Real: {self.result.count('Real')}"
            self.detected_gan_text = f"Detected GAN: {self.result.count('GAN')}"

            self.processed_images_label.setText(self.processed_images_text)
            self.detected_real_label.setText(self.detected_real_text)
            self.detected_gan_label.setText(self.detected_gan_text)
            # Show labels
            self.processed_images_label.show()
            self.detected_real_label.show()
            self.detected_gan_label.show()
        else:
            self.processed_images_label.setText("Processed Images: 0")
            self.detected_real_label.setText("Detected Real: 0")
            self.detected_gan_label.setText("Detected GAN: 0")
            # Show labels
            self.processed_images_label.show()
            self.detected_real_label.show()
            self.detected_gan_label.show()
            messagebox.showinfo(message="No results to display. Please predict results first.") 


    # select model
    def select_model(self):
        self.type = None
        self.notif.show()
        try:
            home_dir = str(Path.home())
            self.model_file, _ = QFileDialog.getOpenFileNames(self, 'Open file', home_dir)
            if ".model" in self.model_file[0]:
                self.loaded_model = load_model(self.model_file[0])
                self.notif.hide()
                messagebox.showinfo(message=f"Model Loaded Successfully")
                print(self.model_file)

                # load trained platt scale (include own directory)
                if "faces" in self.model_file[0]:
                    self.type = "platt scaler/platt_scale_faces.model"
                elif "animals" in self.model_file[0]:
                    self.type = "platt scaler/platt_scale_animals.model"
                elif "objects" in self.model_file[0]:
                    self.type = "platt scaler/platt_scale_objects.model"
                elif "scenes" in self.model_file[0]:
                    self.type ="platt scaler/platt_scale_scenes.model"
                else:
                    self.type = "platt_scale_validate_combined.model"
                    
                # load the clf
                self.clf = load_model(self.type)  
            else:
                self.notif.hide()
                messagebox.showinfo(message="Incompatible model file")
        except:
            print("Something went wrong!")    



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())