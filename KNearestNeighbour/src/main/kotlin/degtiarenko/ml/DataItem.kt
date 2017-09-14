package degtiarenko.ml

class DataItem(x: String, y: String, category: String) {
    val x = x.toDouble()
    val y = y.toDouble()
    val category = category.toInt()
}