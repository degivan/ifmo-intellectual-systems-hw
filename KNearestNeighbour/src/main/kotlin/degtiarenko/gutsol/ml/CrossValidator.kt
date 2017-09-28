package degtiarenko.gutsol.ml

import com.google.common.collect.Lists
import java.util.stream.Collectors

private val FOLD_AMOUNT = 5

class CrossValidator(items: List<DataItem>) {
    private val partition = Lists.partition(items, items.size / FOLD_AMOUNT)

    fun forEachTestSet(consumer: (List<DataItem>, List<DataItem>) -> Unit) {
        for(i in 0 until FOLD_AMOUNT) {
            val trainItems = getTrainItems(i)
            val testItems = getTestItems(i)
            consumer(trainItems, testItems)
        }
    }

    private fun getTestItems(testListIndex: Int): List<DataItem> {
        return partition[testListIndex]
    }

    private fun getTrainItems(testListIndex: Int): List<DataItem> {
        return partition.filterIndexed({ i, _ -> i != testListIndex})
                .stream()
                .flatMap { list -> list.stream() }
                .collect(Collectors.toList())
    }
}