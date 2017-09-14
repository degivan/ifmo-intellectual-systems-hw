package degtiarenko.ml

class Predictor(trainList: List<DataItem>, k: Int, metric: (DataItem, DataItem) -> Double) {
    fun predict(item: DataItem): Int {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    private fun kernelFun(x: Double): Double {
        return if (Math.abs(x) > 1) {
            0.0
        } else {
            1 - Math.abs(x)
        }
    }
}