package degtiarenko.gutsol.ml

import com.sun.xml.internal.fastinfoset.util.StringArray

class ConfigGroup(val spaceTransforms: Iterable<Pair<(DataItem) -> DataItem, String>>,
                  val metrics: Iterable<Pair<(DataItem, DataItem) -> Double, String>>,
                  val kernels: Iterable<Pair<(Double) -> Double, String>>,
                  val kIterable: IntArray) {

    fun fixSpace(spaceName: String): ConfigGroup {
        return ConfigGroup(spaceTransforms.filter { p -> p.second == spaceName },
                metrics, kernels, kIterable)
    }

    fun fixMetric(metricName: String): ConfigGroup {
        return ConfigGroup(spaceTransforms, metrics.filter { p -> p.second == metricName },
                kernels, kIterable)
    }

    fun fixKernel(kernelName: String): ConfigGroup {
        return ConfigGroup(spaceTransforms, metrics, kernels.filter { p -> p.second == kernelName },
                kIterable)
    }

    fun fixK(k: Int): ConfigGroup {
        return ConfigGroup(spaceTransforms, metrics, kernels, kIterable.filter { p -> p == k }.toIntArray())
    }

    fun getPredictors(trainList: List<DataItem>): List<Predictor> {
        val result = mutableListOf<Predictor>()
        for (spaceTransform in spaceTransforms) {
            for (metric in metrics) {
                for (kernel in kernels) {
                    kIterable.mapTo(result) { Predictor(trainList, spaceTransform.first,
                            it, metric.first, kernel.first,
                            arrayListOf(spaceTransform.second, metric.second,kernel.second,it).joinToString()) }
                }
            }
        }
        return result
    }
}