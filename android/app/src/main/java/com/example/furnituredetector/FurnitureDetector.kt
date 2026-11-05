package com.example.furnituredetector

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class Detection(val labelIndex: Int, val score: Float, val box: RectF)
data class Measurement(val preprocessMs: Double, val inferenceMs: Double, val postprocessMs: Double) {
    val totalMs get() = preprocessMs + inferenceMs + postprocessMs
}
data class InferenceResult(val boxes: List<Detection>, val timing: Measurement)

/** Based on YOLO Android/TFLite sample pattern, adapted to a custom 4-class YOLO26 LiteRT model.
 * Accepts FP32 NCHW [1,3,H,W] (observed in user's Colab) or NHWC [1,H,W,3].
 * Supports raw YOLO exports [1,8,N] or [1,N,8], and processed [1,N,6].
 */
class FurnitureDetector(assets: AssetManager) : Closeable {
    val labels = assets.open("labels.txt").bufferedReader().use { it.readLines().filter(String::isNotBlank) }
    private val interpreter: Interpreter
    private val ruLabels = listOf("стул", "стол", "кровать", "диван")
    private val inputWidth: Int
    private val inputNchw: Boolean
    private val inputHeight: Int
    private val outputShape: IntArray
    private val outputBuffer: ByteBuffer
    private val inputBuffer: ByteBuffer

    init {
        require(labels.size == 4) { "Expected four furniture classes in labels.txt" }
        val bytes = assets.open("best.tflite").use { it.readBytes() }
        val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        model.put(bytes); model.rewind()
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(4))
        val input = interpreter.getInputTensor(0)
        val inShape = input.shape()
        require(input.dataType() == DataType.FLOAT32 && inShape.size == 4 && inShape[0] == 1 &&
            (inShape[1] == 3 || inShape[3] == 3)) {
            "Unsupported model input: ${inShape.contentToString()}, ${input.dataType()}; " +
                "expected NCHW [1,3,H,W] or NHWC [1,H,W,3], FLOAT32."
        }
        inputNchw = inShape[1] == 3
        inputHeight = if (inputNchw) inShape[2] else inShape[1]
        inputWidth = if (inputNchw) inShape[3] else inShape[2]
        require(inputHeight > 0 && inputWidth > 0)
        android.util.Log.i("FurnitureYOLO", "Input=${inShape.contentToString()} layout=" +
            (if (inputNchw) "NCHW" else "NHWC"))
        outputShape = interpreter.getOutputTensor(0).shape()
        require(interpreter.getOutputTensor(0).dataType() == DataType.FLOAT32 && outputShape.size == 3 && outputShape[0] == 1) {
            "Unsupported model output: ${outputShape.contentToString()}. Export FP32."
        }
        require(outputShape[1] == 8 || outputShape[2] == 8 || outputShape[2] == 6) {
            "Unsupported output ${outputShape.contentToString()}. Expected [1,8,N], [1,N,8] or [1,N,6]."
        }
        inputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * 4).order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(outputShape.reduce(Int::times) * 4).order(ByteOrder.nativeOrder())
    }

    @Synchronized
    fun infer(bitmap: Bitmap, minConf: Float = 0.25f): InferenceResult {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val scale = min(inputWidth.toFloat() / bitmap.width, inputHeight.toFloat() / bitmap.height)
        val drawW = bitmap.width * scale
        val drawH = bitmap.height * scale
        val padX = (inputWidth - drawW) / 2f
        val padY = (inputHeight - drawH) / 2f
        // RGB floats; letterbox background matches Ultralytics (114/255).
        val resized = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resized)
        canvas.drawColor(Color.rgb(114,114,114))
        canvas.drawBitmap(bitmap, null, RectF(padX, padY, padX + drawW, padY + drawH), Paint(Paint.FILTER_BITMAP_FLAG))
        val pixels = IntArray(inputWidth * inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        inputBuffer.rewind()
        if (inputNchw) {
            // Colab's exported model is [1,3,640,640]: R plane, G plane, B plane.
            // Writing interleaved RGB to this tensor silently breaks predictions.
            for (pix in pixels) inputBuffer.putFloat(Color.red(pix) / 255f)
            for (pix in pixels) inputBuffer.putFloat(Color.green(pix) / 255f)
            for (pix in pixels) inputBuffer.putFloat(Color.blue(pix) / 255f)
        } else {
            for (pix in pixels) {
                inputBuffer.putFloat(Color.red(pix) / 255f)
                inputBuffer.putFloat(Color.green(pix) / 255f)
                inputBuffer.putFloat(Color.blue(pix) / 255f)
            }
        }
        inputBuffer.rewind()
        resized.recycle()
        val t1 = SystemClock.elapsedRealtimeNanos()
        outputBuffer.rewind()
        // Only the model invocation is measured here (teacher's profiling requirement).
        interpreter.run(inputBuffer, outputBuffer)
        val t2 = SystemClock.elapsedRealtimeNanos()
        outputBuffer.rewind()
        val output = FloatArray(outputShape.reduce(Int::times))
        outputBuffer.asFloatBuffer().get(output)
        val detections = decode(output, outputShape, bitmap.width, bitmap.height, padX, padY, scale, minConf)
        val t3 = SystemClock.elapsedRealtimeNanos()
        return InferenceResult(detections, Measurement((t1-t0)/1e6, (t2-t1)/1e6, (t3-t2)/1e6))
    }

    private fun decode(o: FloatArray, sh: IntArray, w: Int, h: Int, px: Float, py: Float,
                       scale: Float, conf: Float): List<Detection> {
        val boxes = mutableListOf<Detection>()
        val processed = sh[2] == 6 && sh[1] != 8
        val channelFirst = sh[1] == 8 && !processed
        val n = if (channelFirst) sh[2] else sh[1]
        val channels = if (processed) 6 else 8
        fun v(k: Int, i: Int): Float = if (channelFirst) o[k*n+i] else o[i*channels+k]
        for (i in 0 until n) {
            if (processed) {
                val score = v(4,i)
                val cls = v(5,i).roundToInt()
                if (!score.isFinite() || score < conf || cls !in 0..3) continue
                val coords = floatArrayOf(v(0,i), v(1,i), v(2,i), v(3,i))
                val normalized = coords.all { it.isFinite() && it in -1f..2f }
                val mulX = if (normalized) inputWidth.toFloat() else 1f
                val mulY = if (normalized) inputHeight.toFloat() else 1f
                addBox(boxes, cls, score, coords[0]*mulX, coords[1]*mulY,
                    coords[2]*mulX, coords[3]*mulY, w,h,px,py,scale)
            } else {
                var best = -1; var score = conf
                for (c in 0..3) {
                    val p = v(4+c, i)
                    if (p.isFinite() && p > score) { score = p; best = c }
                }
                if (best < 0) continue
                var cx = v(0,i); var cy = v(1,i); var bw = v(2,i); var bh = v(3,i)
                if (!cx.isFinite() || !cy.isFinite() || !bw.isFinite() || !bh.isFinite()) continue
                // Some LiteRT exports normalize xywh to 0..1, others retain pixel units.
                if (max(max(kotlin.math.abs(cx),kotlin.math.abs(cy)),max(bw,bh)) <= 2f) {
                    cx *= inputWidth; bw *= inputWidth; cy *= inputHeight; bh *= inputHeight
                }
                addBox(boxes,best,score,cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2,w,h,px,py,scale)
            }
        }
        // Raw one-to-many outputs require class-aware non-maximum suppression.
        return if (processed) boxes.sortedByDescending { it.score }.take(100)
               else nms(boxes,0.7f).take(100)
    }

    private fun addBox(to: MutableList<Detection>, cls: Int, conf: Float,
                       x1: Float,y1: Float,x2: Float,y2: Float,w: Int,h: Int,px: Float,py: Float,s: Float) {
        val left = ((x1-px)/s).coerceIn(0f,w.toFloat())
        val top = ((y1-py)/s).coerceIn(0f,h.toFloat())
        val right = ((x2-px)/s).coerceIn(0f,w.toFloat())
        val bottom = ((y2-py)/s).coerceIn(0f,h.toFloat())
        if (right-left >= 2 && bottom-top >= 2) to += Detection(cls,conf,RectF(left,top,right,bottom))
    }

    private fun nms(source: List<Detection>, threshold: Float): List<Detection> {
        val selected = ArrayList<Detection>()
        for (item in source.sortedByDescending { it.score }) {
            if (selected.size >= 100) break
            if (selected.none { it.labelIndex == item.labelIndex && iou(it.box,item.box) > threshold }) selected.add(item)
        }
        return selected
    }

    private fun iou(a: RectF, b: RectF): Float {
        val ix = max(0f,min(a.right,b.right)-max(a.left,b.left))
        val iy = max(0f,min(a.bottom,b.bottom)-max(a.top,b.top))
        val inter = ix*iy
        val union = a.width()*a.height()+b.width()*b.height()-inter
        return if (union > 0) inter/union else 0f
    }

    fun annotate(bitmap: Bitmap, det: List<Detection>): Bitmap {
        val result = bitmap.copy(Bitmap.Config.ARGB_8888,true)
        val cv = Canvas(result)
        val stroke = max(2f,bitmap.width/200f)
        val textSize = max(20f,bitmap.width/40f)
        val border = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(18,145,210); style = Paint.Style.STROKE; strokeWidth = stroke }
        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; this.textSize=textSize; typeface=android.graphics.Typeface.DEFAULT_BOLD }
        val bg = Paint().apply { color=Color.rgb(10,60,90) }
        for (d in det) {
            cv.drawRect(d.box,border)
            val msg = "${ruLabels[d.labelIndex]} ${String.format(java.util.Locale.US,"%.0f", d.score*100)}%"
            val tw=text.measureText(msg)
            val y = max(textSize+4f, d.box.top)
            cv.drawRect(d.box.left,y-textSize-4f,d.box.left+tw+10f,y+5f,bg)
            cv.drawText(msg,d.box.left+5f,y,text)
        }
        return result
    }

    override fun close() { interpreter.close() }
}
