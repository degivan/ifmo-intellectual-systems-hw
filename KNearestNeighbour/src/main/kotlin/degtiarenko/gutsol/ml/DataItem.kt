package degtiarenko.gutsol.ml

class DataItem(val coords: DoubleArray, val category: Int) {

    constructor(x: String, y: String, category: String): this(doubleArrayOf(x.toDouble(), y.toDouble()),
            category.toInt())
}
