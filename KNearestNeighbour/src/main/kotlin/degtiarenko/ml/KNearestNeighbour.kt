package degtiarenko.ml

import java.util.*


fun main(args: Array<String>) {
    val lines = ClassLoader.getSystemClassLoader().getResource("chips.txt")
            .readText().split("\n").drop(1).dropLast(1)
            .map { s -> s.split(",") }
            .map { l -> DataItem(l[0], l[1], l[2]) }
            .toList()
    Collections.shuffle(lines)

    println(lines[0].x)
}

fun kernelFun(x: Double): Double {
    return if (Math.abs(x) > 1) {
        0.0
    } else {
        1 - Math.abs(x)
    }
}

class DataItem(x: String, y: String, category: String) {
    val x = x.toDouble()
    val y = y.toDouble()
    val category = category.toInt()
}
