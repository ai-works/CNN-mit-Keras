# Da nur der Framework Keras verwendet wird benötigt man ledglich Klassen aus dem KEras Package
# https://keras.io/ Keras Dokumentation

from keras.models import Sequential # Für die Initialisierung und das CNN wird als Sequenz aus Schichten und nicht als Graph gebaut
from keras.layers import Conv2D # die Faltung rein 2D, nur Bilder und keine Videos
from keras.layers import MaxPooling2D # für den Pooling-Schritt, beziehungsweise Dimensionsreduktion und Spatial Indifferenz
from keras.layers import Flatten # Schritt 3 - Flattening, hier werden die pooled feature maps in einen Vektor konvertiert, der anschließend der Input des fully connected NNs sein wird
from keras.layers import Dense # Um die vollvernetzten Schichten in das subsquente NN einzuzuspeisen

### Schritt 1 - Initialisieren des CNN und hinzufügen der ersten bzw. der Input-Schicht
# hierfür erstellen wir ein Objekt der Klasse Sequential
# Größe der Bilder =  64x64 + farbig = rgb
# Anzahl der Filter / Feature Detecter = Anzahl der späteren Feature Maps = 32 = Input-Dimension des nachfolgenden Hidden Layers
# Größe der Filter =  3x3
# Aktivierungsfunktion = relu um nicht-linearität zu gewährleisten und besser ableiten zu können, damit die Gewichte besser aktuallisert werden können
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape(64, 64, 3), activation = "relu"))

### Schritt 2 - MaxPooling Schritt
# Reduzieren der Größer unserer Feature Maps
classifier.add(MaxPooling2D(pool_size= (2,2)))

### Schritt 3 - Zweites Convolutional LAyer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

### Schritt 4 - Zweiter Pooling Schritt
classifier.add(MaxPooling2D(pool_size = (2, 2)))

### Schritt 5 - Flattening um Eingabevektor für das NN zu erhalten
classifier.add(Flatten())

### Schritt 6 Vollvernetztes Hidden Layer hinzufügen
# output_dim = Anzahl der Neuronen im Hidden Layer
# Anzahl der Neuronen im Hidden Layer = keine Daumenregel, aber Zahl zwischen Inout Neuronen und Output-Neuronen
# Anzahl der Neuronen im Hidden Layer = 128
classifier.add(Dense(units = 128, activation = 'relu'))

### Schritt 7 - Outpu Layer hinzufügen
### Anzahl der Output NEuronen nur noch 1 da binäres Problem
### nicht mehr Relu sundern Sigmoid, da wir einen binären Outout haben, Relu geht von 0 bis unendlich, sigmoid geht von 0 bnis 1 = WSK für die jeweilige Klasse
classifier.add(Dense(units = 1, activation = 'sigmoid'))

### Schritt 8 - Kompilieren des CNNs
### Fehlerfunktion = binäre Crossentropy, da binäres Problem
### Adam Optimizer
classifier.complie(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


### Schritt 9 - Data Augmentation mittels Keras' ImageDataGenerator
# Um Overfitting zu verhindern und die Datenmenge künstlich zu erhöhen verwendet man Data Augmentation
# 1. kreiert man viele Batches mit den Bildern
# 2. verwendet man zufällige Transformationen auf eine zufällige Selektion der Bilder in jedem Batch
# Das Modell wird durch die Randomisierung nie das selbe Bild in zwei oder mehr Batches vorfinden
# flow_from_directory = Pfad der Trainings und Testdasten anegeben
# rescale = obligatroisch
# Weiter Data Augmentation Methoden https://keras.io/preprocessing/image/
# Rescale Data auf Werte zwischen 1 und 0
# rescale=1./255 da Pixel Werte zwischen 1 und 255 annehmen können
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# da die trainingsdaten skaliert wurden muss man die Testdaten auch entsprechend skalieren
test_datagen = ImageDataGenerator(rescale = 1./255)

# hier werden die Traingsdaten und Testdaten erstellt
# Der komplette Pfad muss nicht kopiert werden, da selbe Working Directory
# Da die Bilder 64x64 Pixel haben Dimension auf 64x 64 setzen
# Batchsize = Anzahl der Bilder, die durch das CNN laufen, bevor die Gewichte aktualisiert werden
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Hier das selbe mit den Testdaten
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# hier wird das CNN beziehungsweise die Filter trainiert und getestet mit dem Test Set
# Alle vorhandenen Trainingsdaten werden pro Epoche verwendet
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
