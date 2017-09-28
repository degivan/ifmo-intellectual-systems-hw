package degtiarenko.gutsol.ml

import java.lang.Math.*
import java.util.*

val euclideanMetric = { x: DataItem, y: DataItem -> sqrt(pow(x.x - y.x, 2.0) + pow(y.y - x.y, 2.0)) }
val manhattanMetric = { x: DataItem, y: DataItem -> abs(x.x - y.x) + abs(y.y - x.y) }

val metrics = listOf(Pair(euclideanMetric, "euclid"), Pair(manhattanMetric, "manhattan"))

var bestPredictor: Predictor? = null
var bestAccuracy = 0.0

fun main(args: Array<String>) {
    val items = Thread.currentThread().contextClassLoader.getResource("chips.txt")
            .readText().split("\n", "\r")
            .filter { s -> !s.isEmpty() }
            .map { s -> s.split(",") }
            .map { l -> DataItem(l[0], l[1], l[2]) }
            .toList()
    Collections.shuffle(items)

    for ((metricFun, metricName) in metrics) {
        CrossValidator(items).forEachTestSet { trainList, testList ->
            trainWithMetric(trainList, testList, metricFun, metricName)
        }
    }
    val visualizer = Visualizer(items)
    visualizer.drawPlot(bestPredictor as Predictor, "out")
}

fun trainWithMetric(trainList: List<DataItem>, testList: List<DataItem>,
                    metric: (DataItem, DataItem) -> Double, metricName: String): Pair<Double, Int> {
    val results = mutableMapOf<Int, Double>()
    for (k in 1..trainList.size / 2) {
        CrossValidator(trainList).forEachTestSet { trainList2, testList2 ->
            results.putIfAbsent(k, 0.0)
            results.merge(k, trainWithK(trainList2, testList2, metric, k),
                    { acc1, acc2 -> Math.max(acc1, acc2) })
        }
    }
    val goodK = results.maxBy { e -> e.value }!!.key
    val testResult = testWithPredictor(testList, Predictor(trainList, goodK, metric))
    println("Result for metric " + metricName
            + " is: " + testResult
            + " with k: " + goodK)
    if (bestPredictor == null || bestAccuracy < testResult) {
        bestAccuracy = testResult
        bestPredictor = Predictor(trainList, goodK, metric)
    }
    return Pair(testResult, goodK)
}

fun trainWithK(trainList: List<DataItem>, testList: List<DataItem>,
               metric: (DataItem, DataItem) -> Double, k: Int): Double {
    val predictor = Predictor(trainList, k, metric)
    return testWithPredictor(testList, predictor)
}

fun testWithPredictor(testList: List<DataItem>, predictor: Predictor): Double {
    val answers = testList.map { item -> predictor.predict(item) }

    return computeAccuracy(answers, testList)
}

fun computeAccuracy(answers: List<Int>, testList: List<DataItem>): Double {
    val rightAnswers = testList.filterIndexed { i, item -> answers[i] == item.category }
            .size.toDouble()
    return rightAnswers / testList.size
}