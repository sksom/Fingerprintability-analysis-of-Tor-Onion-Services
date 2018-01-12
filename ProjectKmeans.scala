import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

 object ProjectKmeans {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("projkmeans")
    val sc = new SparkContext(sparkConf)

    // Load and parse the data
    val data = sc.textFile("/media/shanky/7E3C4F183C4ECABB/preprocessed_data/preprocessed_data/")
    val parsedData = data.map(s => Vectors.dense(s.split(",",4)(2).toDouble,
      s.split(",",6)(4).toDouble,
      s.split(",",7)(5).substring(1,2).toDouble,
      s.split(",",8)(6).toDouble,
      s.split("[,.]")(9).toDouble,
      s.split("[,.]")(10).toDouble,
      s.split("[,.]")(11).toDouble,
      s.split("[,.]")(12).toDouble,
      s.split("[,.]")(14).toDouble,
      s.split("[,.]")(15).toDouble,
      s.split("[,.]")(16).toDouble,
      s.split("[,.]")(17).toDouble,
      s.split(",")(9).toDouble
    )).cache()     //remove onion_id from a splitted array for training purpose
    val parsedData2 = data.map(s => (s.split(",",2)(0),Vectors.dense(s.split(",",4)(2).toDouble,
      s.split(",",6)(4).toDouble,
      s.split(",",7)(5).substring(1,2).toDouble,
      s.split(",",8)(6).toDouble,
      s.split("[,.]")(9).toDouble,
      s.split("[,.]")(10).toDouble,
      s.split("[,.]")(11).toDouble,
      s.split("[,.]")(12).toDouble,
      s.split("[,.]")(14).toDouble,
      s.split("[,.]")(15).toDouble,
      s.split("[,.]")(16).toDouble,
      s.split("[,.]")(17).toDouble,
      s.split(",")(9).toDouble
    ))).cache()     //make pair (onion_id,corresponding_vector)   for testing purpose

    //parsedData.take(100).foreach(println)

    //Cluster the data into two classes (vulnerable:cluster0 and immune:cluster1) using KMeans
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)    //training the model

    val clusterIdxAndPoint = parsedData2.map(p =>(p._1,clusters.predict(p._2))).sortBy(_._2,false)    //making pair as (onion_id,cluster_id) and sort it on the basis of cluster_id
    //val predictions = clusters.predict(parsedData)     // testing the model
    //printing the results
    //predictions.take(100).foreach(println)
    clusterIdxAndPoint.saveAsTextFile("kmeansOutput")
    //Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

  }
}
