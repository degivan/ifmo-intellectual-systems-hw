package degtiarenko.ml

import java.lang.Math.*
import java.util.*

val euclideanMetric = { x: DataItem, y: DataItem -> sqrt(pow(x.x - y.x, 2.0) + pow(y.y - x.y, 2.0)) }
val manhattanMetric = { x: DataItem, y: DataItem -> abs(x.x - y.x) + abs(y.y - x.y) }

val metrics = listOf(euclideanMetric, manhattanMetric)

fun main(args: Array<String>) {
    val items = ClassLoader.getSystemClassLoader().getResource("chips.txt")
            .readText().split("\n").drop(1).dropLast(1)
            .map { s -> s.split(",") }
            .map { l -> DataItem(l[0], l[1], l[2]) }
            .toList()
    Collections.shuffle(items)

    CrossValidator(items).forEachTestSet { trainList, testList -> /*do something useful*/ }
}

fun kernelFun(x: Double): Double {
    return if (Math.abs(x) > 1) {
        0.0
    } else {
        1 - Math.abs(x)
    }
}
