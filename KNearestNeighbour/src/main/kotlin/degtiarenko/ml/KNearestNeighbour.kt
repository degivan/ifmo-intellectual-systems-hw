package degtiarenko.ml

import java.lang.Math.*
import java.util.*

val euclideanMetric = { x: DataItem, y: DataItem -> sqrt(pow(x.x - y.x, 2.0) + pow(y.y - x.y, 2.0)) }
val manhattanMetric = { x: DataItem, y: DataItem -> abs(x.x - y.x) + abs(y.y - x.y) }

val metrics = listOf(euclideanMetric, manhattanMetric)

fun main(args: Array<String>) {
    val items = Thread.currentThread().contextClassLoader.getResource("chips.txt")
            .readText().split("\n").drop(1).dropLast(1)
            .map { s -> s.split(",") }
            .map { l -> DataItem(l[0], l[1], l[2]) }
            .toList()
    Collections.shuffle(items)

    for (metric in metrics) {
        CrossValidator(items).forEachTestSet { trainList, testList ->
            trainWithMetric(trainList, testList, metric)
        }
    }
}

fun trainWithMetric(trainList: List<DataItem>, testList: List<DataItem>,
                    metric: (DataItem, DataItem) -> Double) {
    val results = mutableMapOf<Int, Double>()
    for (k in 1..trainList.size / 2) {
        CrossValidator(trainList).forEachTestSet { trainList2, testList2 ->
            results.put(k, trainWithK(trainList2, testList2, metric, k))
        }
    }
    val goodK = results.maxBy { e -> e.value }!!.key
    val testResult = testWithPredictor(testList, Predictor(trainList, goodK, metric))
    println("Result for metric " + metric.toString()
            + " is: " + testResult
            + " with k: " + goodK)
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
    var rightAnswers = 0.0
    for (i in 0..testList.size - 1) {
        if (answers[i] == testList[i].category) {
            rightAnswers++
        }
    }
    return rightAnswers / testList.size
}
