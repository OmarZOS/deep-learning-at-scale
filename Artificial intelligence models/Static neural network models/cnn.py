# """Test convolutional model with Keras and Spark"""
# Usage :  python3 *columnNames *columnTypes targetColumn storagePath




import numpy as np
import random
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD
import theano.tensor as T


from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, IntegerType, StringType ,FloatType

# # To read arguments 
import sys


def mae_clip(y_true, y_pred):
    """Return the MAE with clipping to provide resistance to outliers"""
    CLIP_VALUE = 6
    return T.clip(T.abs_(y_true - y_pred), 0, CLIP_VALUE).mean(axis=-1)

# # Seed random number generator
# np.random.seed(42)


# Model name
MDL_NAME = "cnn"

columnNames = sys.argv[1] 
columnTypes = sys.argv[2] 
targetColumn = sys.argv[3]
storagePath= sys.argv[4]



# # Prior, to doing anything else, we need to initialize a Spark session.
spark = SparkSession.builder.appName("CNN prediction").getOrCreate()


# #receiving the arguments, whether integers, floats or strings..
schema = StructType([StructField(k,
                                 (((StringType(),
                                    FloatType())
                                   [v=="float"]
                                   ,IntegerType())
                                  [v=="int"]),True) 
                     for (k,v) in list(zip(columnNames.split(',')
                                           ,columnTypes.split(',')) ) ])
                                 
print(schema) #lol


dataset = spark.read.csv(sys.argv[2], header=False, schema=schema)

# Split into 3D datasets
train_data, validation_data, test_dataset = dataset.randomSplit([0.64,0.16,0.2])


#indexing categorical variables
categorical_variables = [ k for (k,v) in list(zip(columnNames.split(','),columnTypes.split(','))) if v != "int" and v!="float" ]
print("Categorical variables: "+categorical_variables)

indexers = [StringIndexer(inputCol=column, outputCol=column+"-index") for column in categorical_variables]

encoder = OneHotEncoder(
    inputCols=[indexer.getOutputCol() for indexer in indexers],
    outputCols=["{0}-encoded".format(indexer.getOutputCol()) for indexer in indexers]
)

assembler = VectorAssembler(
    inputCols=encoder.getOutputCols(),
    outputCol="categorical-features"
)

pipeline = Pipeline(stages=indexers + [encoder, assembler])

train_data = pipeline.fit(train_data).transform(train_data)
validation_data = pipeline.fit(validation_data).transform(validation_data)
test_dataset = pipeline.fit(test_dataset).transform(test_dataset)

continuous_variables = [ k for (k,v) in list(zip(columnNames.split(','),columnTypes.split(','))) if v == "int" or v=="float" ]

#combining these continuous variables with those that are categorical
assembler = VectorAssembler(
    inputCols=['categorical-features', *continuous_variables],
    outputCol='features'
)

train_data = assembler.transform(train_data)
validation_data = assembler.transform(validation_data)
test_dataset = assembler.transform(test_dataset)


# Finally, we encode our target label.
indexer = StringIndexer(inputCol=targetColumn, outputCol='label')

# Build neural network
model = Sequential()
model.add(Convolution1D(1, 100, 13, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(3600, 12))

# Use stochastic gradient descent and compile model
sgd = SGD(lr=0.001, momentum=0.99, decay=1e-6, nesterov=True)
model.compile(loss=mae_clip, optimizer=sgd)

# Use early stopping and saving as callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10)
save_best = ModelCheckpoint(storagePath+"models/%s.mdl" % MDL_NAME, save_best_only=True)
callbacks = [early_stop, save_best]

# Train model
t0 = time.time()
hist = model.fit(featuresCol='features', validation_data=validation_data,
                 verbose=2, callbacks=callbacks, nb_epoch=1000, batch_size=20)
time_elapsed = time.time() - t0

# Print time elapsed and loss on testing dataset
test_set_loss = model.test_on_batch(test_dataset)
print ("\nTime elapsed: %f s") % time_elapsed
print ("Testing set loss: %f") % test_set_loss







