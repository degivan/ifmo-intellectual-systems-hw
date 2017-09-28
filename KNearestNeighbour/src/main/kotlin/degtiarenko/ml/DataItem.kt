package degtiarenko.ml

class DataItem(val x: Double, val y: Double, val category: Int) {

    constructor(x: String, y: String, category: String): this(x.toDouble(), y.toDouble(), category.toInt())
}
