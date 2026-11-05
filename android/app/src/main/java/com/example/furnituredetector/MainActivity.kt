package com.example.furnituredetector

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.documentfile.provider.DocumentFile
import java.util.Locale
import java.util.concurrent.Executors

/** Choose a phone folder via Android Storage Access Framework, then detect each image locally. */
class MainActivity : Activity() {
    private val PICK_FOLDER = 10
    private val PICK_IMAGE = 11
    private val SAVE_CSV = 12
    private val worker = Executors.newSingleThreadExecutor()
    private val files = mutableListOf<Uri>()
    private val reports = mutableListOf<Report>()
    private lateinit var image: ImageView
    private lateinit var info: TextView
    private var detector: FurnitureDetector? = null
    private var position = 0
    private var busy = false

    private data class Report(val uri: Uri, val boxes: Int, val timing: Measurement,
                              val mode: String, val runIndex: Int = 0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(22,20,22,20) }
        val scroller = ScrollView(this)
        scroller.addView(root)
        setContentView(scroller)
        root.addView(TextView(this).apply { text="YOLO26n — мебель"; textSize=24f; gravity=Gravity.CENTER })
        root.addView(TextView(this).apply { text="Классы: стул, стол, кровать, диван. Анализ полностью на устройстве."; textSize=15f })
        val firstRow=LinearLayout(this).apply { orientation=LinearLayout.HORIZONTAL }
        root.addView(firstRow)
        firstRow.addView(button("Выбрать папку") { pickFolder() }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT,1f))
        firstRow.addView(button("Одно фото") { pickImage() }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT,1f))
        val secondRow=LinearLayout(this).apply { orientation=LinearLayout.HORIZONTAL }
        root.addView(secondRow)
        secondRow.addView(button("Следующее") { nextImage() }, LinearLayout.LayoutParams(0,ViewGroup.LayoutParams.WRAP_CONTENT,1f))
        secondRow.addView(button("Обработать всё") { processAll() }, LinearLayout.LayoutParams(0,ViewGroup.LayoutParams.WRAP_CONTENT,1f))
        root.addView(button("Профилирование (12 запусков)") { benchmark() })
        root.addView(button("Сохранить CSV с временем") { exportCsv() })
        info=TextView(this).apply { textSize=15f; setPadding(0,12,0,12) }
        root.addView(info)
        image=ImageView(this).apply { adjustViewBounds=true; scaleType=ImageView.ScaleType.FIT_CENTER }
        root.addView(image,LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        info.text=if (assets.list("")?.contains("best.tflite") == true) "Модель найдена. Выберите фото или папку." else
            "Модель best.tflite пока отсутствует в assets. Сначала обучите модель в Colab и экспортируйте её."
    }

    private fun button(title:String, handler:()->Unit)=Button(this).apply { text=title; setOnClickListener { if (!busy) handler() } }
    private fun pickFolder() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        startActivityForResult(intent,PICK_FOLDER)
    }
    private fun pickImage() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply { type="image/*"; addCategory(Intent.CATEGORY_OPENABLE) }
        startActivityForResult(intent,PICK_IMAGE)
    }
    private fun exportCsv() {
        if (reports.isEmpty()) { info.text="Сначала обработайте изображения."; return }
        val intent=Intent(Intent.ACTION_CREATE_DOCUMENT).apply { addCategory(Intent.CATEGORY_OPENABLE); type="text/csv"; putExtra(Intent.EXTRA_TITLE,"yolo26_android_profile.csv") }
        startActivityForResult(intent,SAVE_CSV)
    }
    @Deprecated("Legacy SAF activity result API")
    override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?) {
        super.onActivityResult(requestCode,resultCode,data)
        if (resultCode!=RESULT_OK || data?.data==null) return
        val uri=data.data!!
        when(requestCode) {
            PICK_FOLDER -> {
                try { contentResolver.takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: Exception) { }
                files.clear(); reports.clear(); position=0
                val parent=DocumentFile.fromTreeUri(this,uri)
                if(parent!=null) collectImages(parent,files)
                files.sortBy { it.toString() }
                info.text="Найдено ${files.size} изображений. Следующее — обработает первое фото, «Обработать всё» — все."
                if(files.isNotEmpty()) processCurrent()
            }
            PICK_IMAGE -> { files.clear();files+=uri;position=0;reports.clear();processCurrent() }
            SAVE_CSV -> saveCsv(uri)
        }
    }
    private fun collectImages(dir:DocumentFile,out:MutableList<Uri>) {
        if(out.size>=500) return
        for(f in dir.listFiles()) {
            if(f.isDirectory) collectImages(f,out)
            else if(f.isFile && (f.type?.startsWith("image/")==true ||
                        listOf(".jpg",".jpeg",".png",".webp",".bmp").any { (f.name?:"").lowercase().endsWith(it) })) out+=f.uri
            if(out.size>=500) break
        }
    }
    private fun bitmap(uri:Uri):Bitmap {
        val opt=BitmapFactory.Options().apply { inJustDecodeBounds=true }
        contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it,null,opt) }
        var sample=1
        while(opt.outWidth/sample>1600 || opt.outHeight/sample>1600) sample*=2
        val decode=BitmapFactory.Options().apply { inSampleSize=sample; inPreferredConfig=Bitmap.Config.ARGB_8888 }
        return contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it,null,decode) }
            ?: error("Failed to decode image: $uri")
    }
    private fun nextImage() {
        if(files.isEmpty()){info.text="Выберите папку или фото.";return}
        position=(position+1)%files.size
        processCurrent()
    }
    private fun getDetector():FurnitureDetector {
        var d=detector
        if(d==null){d=FurnitureDetector(assets);detector=d}
        return d
    }
    private fun processCurrent() {
        if(files.isEmpty())return
        busy=true
        info.text="Идёт обработка ${position+1}/${files.size}..."
        val uri=files[position]
        worker.execute {
            try {
                val src=bitmap(uri)
                val d=getDetector()
                val result=d.infer(src)
                val painted=d.annotate(src,result.boxes)
                synchronized(reports){reports+=Report(uri,result.boxes.size,result.timing,"single")}
                runOnUiThread {
                    image.setImageBitmap(painted)
                    info.text="Фото ${position+1}/${files.size}; найдено: ${result.boxes.size}\n"+
                        "Модель: ${ms(result.timing.inferenceMs)} мс; подготовка: ${ms(result.timing.preprocessMs)} мс; "+
                        "постобработка: ${ms(result.timing.postprocessMs)} мс; всего: ${ms(result.timing.totalMs)} мс"
                    busy=false
                }
            }catch(e:Exception){fail(e)}
        }
    }
    private fun processAll() {
        if(files.isEmpty()){info.text="Сначала выберите папку с фото.";return}
        busy=true
        reports.clear()
        info.text="Пакетная обработка ${files.size} изображений..."
        val snapshot=files.toList()
        worker.execute {
            try {
                val d=getDetector()
                var preview:Bitmap?=null
                snapshot.forEachIndexed { index, uri ->
                    val src=bitmap(uri)
                    val result=d.infer(src)
                    synchronized(reports){reports+=Report(uri,result.boxes.size,result.timing,"batch")}
                    if(index==snapshot.lastIndex) preview=d.annotate(src,result.boxes)
                }
                val timings=synchronized(reports){reports.map { it.timing.inferenceMs }}
                val lastPreview=preview
                runOnUiThread {
                    if(lastPreview!=null)image.setImageBitmap(lastPreview)
                    info.text="Обработано ${timings.size} изображений. Среднее время model.run: ${ms(timings.average())} мс. "+
                        "Минимум ${ms(timings.minOrNull()?:0.0)}, максимум ${ms(timings.maxOrNull()?:0.0)} мс. "+
                        "Нажмите «Сохранить CSV с временем»."
                    busy=false
                }
            }catch(e:Exception){fail(e)}
        }
    }
    private fun benchmark() {
        if(files.isEmpty()){info.text="Сначала выберите фото.";return}
        busy=true
        val uri=files[position]
        info.text="Прогрев 2 раза, затем 10 замеров только interpreter.run()..."
        worker.execute {
            try {
                val d=getDetector(); val src=bitmap(uri)
                repeat(2){d.infer(src)}
                val measured=(1..10).map { index ->
                    val result=d.infer(src)
                    synchronized(reports) {
                        reports += Report(uri, result.boxes.size, result.timing, "benchmark", index)
                    }
                    result.timing.inferenceMs
                }
                val times=measured
                val sorted=times.sorted()
                runOnUiThread {
                    info.text="10 запусков после 2 прогревов (сохранены в CSV). Инференс: среднее ${ms(times.average())} мс, "+
                        "медиана ${ms((sorted[4]+sorted[5])/2)} мс, "+
                        "мин ${ms(times.minOrNull()?:0.0)} / макс ${ms(times.maxOrNull()?:0.0)} мс. "+
                        "Оценка 1000/среднее: ${ms(1000/times.average())} вызовов/с (не FPS камеры)."
                    busy=false
                }
            }catch(e:Exception){fail(e)}
        }
    }
    private fun saveCsv(uri:Uri) {
        try {
            val data=synchronized(reports){reports.toList()}
            contentResolver.openOutputStream(uri)?.bufferedWriter()?.use {w ->
                w.appendLine("uri,mode,run_index,detections,preprocess_ms,inference_ms,postprocess_ms,total_ms")
                data.forEach {r ->
                    w.appendLine("\"${r.uri.toString().replace("\"","\"\"")}\",${r.mode},${r.runIndex},${r.boxes},"+
                        "${ms(r.timing.preprocessMs)},${ms(r.timing.inferenceMs)},${ms(r.timing.postprocessMs)},${ms(r.timing.totalMs)}")
                }
            } ?: error("Cannot open CSV output URI")
            info.text="CSV сохранён (${data.size} строк)."
        } catch(e:Exception){info.text="Ошибка сохранения CSV: ${e.message}"}
    }
    private fun ms(v:Double)=String.format(Locale.US,"%.2f",v)
    private fun fail(e:Exception) {
        android.util.Log.e("FurnitureYOLO", "Inference error",e)
        runOnUiThread { info.text="Ошибка: ${e.message}. Проверьте модель best.tflite, формат входа и выходные тензоры.";busy=false }
    }
    override fun onDestroy(){super.onDestroy();worker.shutdown();detector?.close()}
}
