package degtiarenko.ml


fun main(args: Array<String>) {
    var lines = ClassLoader.getSystemClassLoader().getResource("chips.txt")
            .readText().split("\n").toMutableList()
    var datasetSize = lines[0].toInt()
    lines.removeAt(0)

    println(kernelFun(2.0))
    println(kernelFun(0.5))
    println(kernelFun(0.3))
}

fun kernelFun(x: Double): Double {
    return if (Math.abs(x) > 1) {
        0.0
    } else {
        1 - Math.abs(x)
    }
}
