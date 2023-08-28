from PyQt6 import QtCore, QtGui, QtWidgets
import sys

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(900, 582)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.home(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

    def home(self, Form):
        Form.setStyleSheet("#Form {\n""background-image: url(assets/Get Started.jpg);\n""}")

        self.label = QtWidgets.QLabel(parent=Form)
        self.getStarted = QtWidgets.QPushButton(parent=Form)
        self.login = QtWidgets.QPushButton(parent=Form)
        self.label_2 = QtWidgets.QLabel(parent=Form)
        self.label_3 = QtWidgets.QLabel(parent=Form)
        self.label_4 = QtWidgets.QLabel(parent=Form)
        self.label_5 = QtWidgets.QLabel(parent=Form)

        self.label.setGeometry(QtCore.QRect(330, 310, 241, 231))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("assets/Icon.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.getStarted.setGeometry(QtCore.QRect(330, 170, 131, 61))
        self.getStarted.setAutoFillBackground(False)
        self.getStarted.setStyleSheet("*{\n""    background: transparent\n""}")
        self.getStarted.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("assets/Button1.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.getStarted.setIcon(icon)
        self.getStarted.setIconSize(QtCore.QSize(105, 105))
        self.getStarted.setObjectName("getStarted")

        self.getStarted.clicked.connect(lambda:self.about(Form))

        self.login.setGeometry(QtCore.QRect(450, 170, 131, 61))
        self.login.setAutoFillBackground(False)
        self.login.setStyleSheet("*{\n""background: transparent\n""}")
        self.login.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("assets/Button2.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.login.setIcon(icon1)
        self.login.setIconSize(QtCore.QSize(105, 105))
        self.login.setObjectName("login")

        self.label_2.setGeometry(QtCore.QRect(590, 310, 131, 131))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("assets/eye.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")

        self.label_3.setGeometry(QtCore.QRect(600, 440, 121, 101))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap("assets/gallery1.png"))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName("label_3")

        self.label_4.setGeometry(QtCore.QRect(190, 320, 131, 131))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("assets/gallery.png"))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")

        self.label_5.setGeometry(QtCore.QRect(190, 450, 131, 91))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap("assets/eye1.png"))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName("label_5")

    def about(self, Form):
        Form.setObjectName("Form")
        Form.resize(900, 582)
        Form.setStyleSheet("#Form {\n""background-image: url(assets/About.png);\n""}")






            


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(form)
    form.show()
    sys.exit(app.exec())