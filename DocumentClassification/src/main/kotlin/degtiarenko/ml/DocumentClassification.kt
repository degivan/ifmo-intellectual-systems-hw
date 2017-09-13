package degtiarenko.ml

fun main(args : Array<String>) {
    val lines = ClassLoader.getSystemClassLoader().getResource("trainingdata.txt")
            .readText().split("\n")
    var datasetSize = lines[0].toInt()

    println(kernelFun(2.0))
    println(kernelFun(0.5))
    println(kernelFun(0.3))
}

fun kernelFun(x : Double) : Double {
    return if (Math.abs(x) > 1) {
        0.0
    } else {
        1 - Math.abs(x)
    }
}
