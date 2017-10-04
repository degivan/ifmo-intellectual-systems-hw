package degtiarenko.gutsol.ml

import de.erichseifert.gral.data.DataSeries
import de.erichseifert.gral.data.DataTable
import de.erichseifert.gral.plots.XYPlot
import java.io.File
import de.erichseifert.gral.io.plots.DrawableWriterFactory

import java.util.*
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D
import de.erichseifert.gral.plots.points.PointRenderer
import java.awt.Color
import java.awt.geom.Ellipse2D

/**
 * Created by ksenia on 27.09.17.
 */
val rnd = Random(0)
const val POINTS_COUNT = 100000
const val MIN_X = -1.1
const val MIN_Y = -1.1
const val WIDTH = 2.2
const val HEIGHT = 2.2

val blue = getPointRenderer(Color.BLUE)
val yellow = getPointRenderer(Color.YELLOW)
val darkBlue = getPointRenderer(Color.decode("#0B0B61"))
val darkYellow = getPointRenderer(Color.decode("#FF8000"))

class Visualizer(trainSet: List<DataItem>) {
    val blueSamples: DataSeries
    val yellowSamples: DataSeries

    init {
        val blueSamplesData = DataTable(Double::class.javaObjectType, Double::class.javaObjectType)
        val yellowSamplesData = DataTable(Double::class.javaObjectType, Double::class.javaObjectType)
        for (dataItem in trainSet) {
            if (dataItem.category == 0) {
                blueSamplesData.add(dataItem.coords[0], dataItem.coords[1])
            } else {
                yellowSamplesData.add(dataItem.coords[0], dataItem.coords[1])
            }
        }
        blueSamples = DataSeries(blueSamplesData)
        yellowSamples = DataSeries(yellowSamplesData)
    }

    fun drawPlot(predictor: Predictor, outputFileName: String) {
        val yellowData = DataTable(Double::class.javaObjectType, Double::class.javaObjectType)
        val blueData = DataTable(Double::class.javaObjectType, Double::class.javaObjectType)
        for (i in 0..POINTS_COUNT) {
            val x = MIN_X + rnd.nextDouble() * WIDTH
            val y = MIN_Y + rnd.nextDouble() * HEIGHT
            val dataItem = DataItem(doubleArrayOf(x, y), 1)
            if (predictor.predict(dataItem) > 0) {
                yellowData.add(x, y)
            } else {
                blueData.add(x, y)
            }
        }
        val yellowPoints = DataSeries(yellowData)
        val bluePoints = DataSeries(blueData)
        val plot = XYPlot(bluePoints, yellowPoints, blueSamples, yellowSamples)
        plot.setBounds(MIN_X, MIN_Y, WIDTH, HEIGHT)
        plot.setPointRenderers(bluePoints, blue)
        plot.setPointRenderers(yellowPoints, yellow)
        plot.setPointRenderers(blueSamples, darkBlue)
        plot.setPointRenderers(yellowSamples, darkYellow)
        val writer = DrawableWriterFactory.getInstance().get("image/png")
        writer.write(plot, File(outputFileName + ".png").outputStream(), 800.0, 800.0)
    }
}

private fun getPointRenderer(color: Color): PointRenderer {
    val renderer = DefaultPointRenderer2D()
    renderer.shape = Ellipse2D.Double(-3.0, -3.0, 12.0, 12.0)
    renderer.setColor(color)
    return renderer
}