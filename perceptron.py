from numpy import array
from math import sqrt
from pyspark import SparkContext
from pyspark import SparkConf
from py4j.compat import unicode
from py4j.java_gateway import JavaGateway
from py4j.protocol import Py4JJavaError, Py4JError
from py4j.tests.java_gateway_test import PY4J_JAVA_PATH, safe_shutdown
from pyspark.sql import Row
from pyspark.rdd import RDD
from pyspark.sql.types import *
from pyspark.sql.types import StringType
from pyspark.sql import SQLContext
from pyspark.ml.feature import IndexToString, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

conf = SparkConf()
conf.setMaster("local")
conf.setAppName('perceptron')
#spark = SparkSession.builder.getOrCreate()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
data = sc.textFile('data_1')
data = data.map(lambda x: x.split(","))
data, test_data = data.randomSplit([0.7,0.3])
data = data.map(lambda p: Row(
    ser_name = (p[0].split(".")[0]).split("_")[1],
    time = p[1],
    ttl = p[2],
    p_id = p[3],
    offset = p[4],
    tcp_type = p[5],
    p_length = p[6],
    source_ip = p[7],
    dest_ip = p[8],
    tcp_portno = p[9]))
test_data = test_data.map(lambda p: Row(
    ser_name = (p[0].split(".")[0]).split("_")[1],
    time = p[1],
    ttl = p[2],
    p_id = p[3],
    offset = p[4],
    tcp_type = p[5],
    p_length = p[6],
    source_ip = p[7],
    dest_ip = p[8],
    tcp_portno = p[9]))
data_DF = sqlContext.createDataFrame(data)
indexer = StringIndexer(inputCol="ser_name", outputCol="label")
model = indexer.fit(data_DF)
indexed = model.transform(data_DF)
indexer = StringIndexer(inputCol="source_ip", outputCol="source_ipIndex")
model = indexer.fit(indexed)
indexed = model.transform(indexed)
indexer = StringIndexer(inputCol="dest_ip", outputCol="dest_ipIndex")
model = indexer.fit(indexed)
indexed = model.transform(indexed)
indexer = StringIndexer(inputCol="tcp_type", outputCol="tcp_typeIndex")
model = indexer.fit(indexed)
indexed = model.transform(indexed)
indexer = StringIndexer(inputCol="tcp_portno", outputCol="tcp_portnoIndex")
model = indexer.fit(indexed)
indexed = model.transform(indexed)
req_dataDF = indexed.select("label","ttl","offset","tcp_typeIndex","p_length","source_ipIndex","dest_ipIndex","tcp_portnoIndex")



test_data_DF = sqlContext.createDataFrame(test_data)
indexer = StringIndexer(inputCol="ser_name", outputCol="label")
model = indexer.fit(test_data_DF)
indexed = model.transform(test_data_DF)
indexer = StringIndexer(inputCol="source_ip", outputCol="source_ipIndex")
model = indexer.fit(indexed)
indexed = model.transform(indexed)
indexer = StringIndexer(inputCol="dest_ip", outputCol="dest_ipIndex")
model = indexer.fit(indexed)
indexed = model.transform(indexed)
indexer = StringIndexer(inputCol="tcp_type",outputCol="tcp_typeIndex")
model = indexer.fit(indexed)
indexed = model.transform(indexed)
indexer = StringIndexer(inputCol="tcp_portno",outputCol="tcp_portnoIndex")
model = indexer.fit(indexed)
indexed = model.transform(indexed)
req_test_dataDF = indexed.select("label","ttl","offset","tcp_typeIndex","p_length","source_ipIndex","dest_ipIndex","tcp_portnoIndex")

assembler = VectorAssembler(
  inputCols=["ttl","offset","tcp_typeIndex","p_length","source_ipIndex","dest_ipIndex","tcp_portnoIndex"], outputCol="features"
)
expr = [col(c).cast("Double").alias(c) 
        for c in assembler.getInputCols()]



layers = [7, 2, 1, 3]



df2 = req_dataDF.select("label",*expr)
df = assembler.transform(df2.na.drop())
training = df.select("label","features")
test_df2 = req_test_dataDF.select("label",*expr)
test_df = assembler.transform(test_df2.na.drop())
testing = test_df.select("label","features")
#lr = LinearSVC(maxIter=10, regParam=0.1)

trainer = MultilayerPerceptronClassifier(maxIter=2, layers=layers, blockSize=128, seed=1234)


#print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")
lrModel = trainer.fit(training)
#print("Coefficients: \n" + str(lrModel.coefficientMatrix))
#print("Intercept: " + str(lrModel.interceptVector))
prediction = lrModel.transform(testing)
result = prediction.select("prediction", "label") \
    .collect()
count =0.0



#evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
#print("Test set accuracy = " + str(evaluator.evaluate(result)))


for row in result:
    #print("features=%s, label=%s -> prob=%s, prediction=%s"
    #      % (row.features, row.label, row.probability, row.prediction))
    if (float(row.label)==float(row.prediction)):
        count = count+1
print ("accuracy:"+str(count/testing.count()))
