package degtiarenko.gutsol.ml

class Predictor(trainList: List<DataItem>, val spaceTransform: (DataItem) -> DataItem,
                val k: Int, val metric: (DataItem, DataItem) -> Double,
                val kernel: (Double) -> Double, val name: String) {
    private val trainList: List<DataItem> = trainList.map(spaceTransform)

    fun predict(oldItem: DataItem): Int {
        val item = spaceTransform(oldItem)
        val neighbours = trainList.sortedBy { trainItem -> metric(trainItem, item) }.take(k)
        val kNeighbour = neighbours[k - 1]
        val categoryWeights = mutableMapOf<Int, Double>()
        for (neighbour in neighbours) {
            categoryWeights.computeIfAbsent(neighbour.category, { 0.0 })
            categoryWeights.merge(neighbour.category, weight(item, neighbour, kNeighbour),
                    { w1, w2 -> w1 + w2 })
        }
        return categoryWeights.entries.maxBy { e -> e.value }!!.key
    }

    private fun weight(item: DataItem, neighbour: DataItem, kNeighbour: DataItem): Double {
        return kernel(metric(item, neighbour) / metric(item, kNeighbour))
    }

    private fun kernelFun(x: Double): Double {
        return if (Math.abs(x) > 1) {
            0.0
        } else {
            1.0 //- Math.abs(coords) TODO:померить,что лучше
        }
    }

}