package com.ipdds.plantdisease

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.Intent
import android.graphics.Color
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import androidx.drawerlayout.widget.DrawerLayout
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileNotFoundException
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

class MainActivity : Activity() {
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()
    private val pageHistory = ArrayDeque<View>()
    private var diseaseInterpreter: Interpreter? = null
    private var cropInterpreter: Interpreter? = null
    private var selectedImage: Uri? = null
    private var classNames: List<String> = emptyList()
    private var classMetadata: JSONObject? = null
    private var modelReady = false
    private var pendingCameraFile: File? = null
    private var selectedCameraFile: File? = null
    private val imageSize = 128

    private lateinit var drawerLayout: DrawerLayout
    private lateinit var drawerPanel: View
    private lateinit var mainPage: View
    private lateinit var aboutPage: View
    private lateinit var settingsPage: View
    private lateinit var preview: ImageView
    private lateinit var emptyImageHint: View
    private lateinit var selectedFile: TextView
    private lateinit var cropSelector: Spinner
    private lateinit var analyzeButton: Button
    private lateinit var progress: ProgressBar
    private lateinit var status: TextView
    private lateinit var resultPanel: View
    private lateinit var resultCrop: TextView
    private lateinit var resultDiagnosis: TextView
    private lateinit var resultConfidence: TextView
    private lateinit var resultPathogen: TextView
    private lateinit var resultTreatment: TextView
    private lateinit var resultWarning: TextView
    private lateinit var rememberPlantSelection: Switch
    private lateinit var capturePhotoButton: View

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setupNavigation()
        connectViews()
        cropSelector.adapter = ArrayAdapter.createFromResource(
            this,
            R.array.crop_options,
            android.R.layout.simple_spinner_item
        ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
        setupPlantSelection()

        findViewById<Button>(R.id.choose_image).setOnClickListener { chooseImage() }
        capturePhotoButton = findViewById(R.id.capture_photo)
        capturePhotoButton.setOnClickListener { capturePhoto() }
        analyzeButton.setOnClickListener { analyzeImage() }

        try {
            loadMetadata()
            loadModels()
            modelReady = true
        } catch (error: Exception) {
            showStatus("The on-device models could not be loaded: ${error.message ?: "unknown error"}")
            analyzeButton.isEnabled = false
        }
    }

    private fun setupNavigation() {
        drawerLayout = findViewById(R.id.drawer_layout)
        drawerPanel = findViewById(R.id.navigation_drawer)
        mainPage = findViewById(R.id.main_page)
        aboutPage = findViewById(R.id.about_page)
        settingsPage = findViewById(R.id.settings_page)

        val openDrawer = View.OnClickListener { drawerLayout.openDrawer(drawerPanel) }
        findViewById<View>(R.id.open_drawer).setOnClickListener(openDrawer)
        findViewById<View>(R.id.about_open_drawer).setOnClickListener(openDrawer)
        findViewById<View>(R.id.settings_open_drawer).setOnClickListener(openDrawer)
        findViewById<View>(R.id.about_back).setOnClickListener { navigateBack() }
        findViewById<View>(R.id.settings_back).setOnClickListener { navigateBack() }
        findViewById<View>(R.id.nav_leaf_checker).setOnClickListener { showPage(mainPage) }
        findViewById<View>(R.id.nav_about).setOnClickListener { showPage(aboutPage) }
        findViewById<View>(R.id.nav_settings).setOnClickListener { showPage(settingsPage) }
    }

    private fun setupPlantSelection() {
        rememberPlantSelection = findViewById(R.id.remember_plant_selection)
        val preferences = getSharedPreferences(SETTINGS_PREFERENCES, MODE_PRIVATE)
        val shouldRemember = preferences.getBoolean(KEY_REMEMBER_PLANT_SELECTION, false)
        rememberPlantSelection.isChecked = shouldRemember
        cropSelector.setSelection(
            if (shouldRemember) {
                preferences.getInt(KEY_LAST_PLANT_SELECTION, 0).coerceIn(0, 2)
            } else {
                0
            }
        )
        cropSelector.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (rememberPlantSelection.isChecked) {
                    preferences.edit().putInt(KEY_LAST_PLANT_SELECTION, position).apply()
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        rememberPlantSelection.setOnCheckedChangeListener { _, isChecked ->
            preferences.edit().apply {
                putBoolean(KEY_REMEMBER_PLANT_SELECTION, isChecked)
                if (isChecked) {
                    putInt(KEY_LAST_PLANT_SELECTION, cropSelector.selectedItemPosition)
                } else {
                    remove(KEY_LAST_PLANT_SELECTION)
                }
            }.apply()
        }
    }

    private fun showPage(selectedPage: View) {
        val currentPage = listOf(mainPage, aboutPage, settingsPage)
            .firstOrNull { it.visibility == View.VISIBLE }
        if (currentPage != null && currentPage !== selectedPage) {
            pageHistory.addLast(currentPage)
        }
        displayPage(selectedPage)
    }

    private fun displayPage(selectedPage: View) {
        listOf(mainPage, aboutPage, settingsPage).forEach { page ->
            page.visibility = if (page === selectedPage) View.VISIBLE else View.GONE
        }
        drawerLayout.closeDrawer(drawerPanel)
    }

    private fun navigateBack() {
        val previousPage = if (pageHistory.isEmpty()) mainPage else pageHistory.removeLast()
        displayPage(previousPage)
    }

    @Deprecated("Use the system back dispatcher.")
    @Suppress("DEPRECATION")
    override fun onBackPressed() {
        when {
            drawerLayout.isDrawerOpen(drawerPanel) -> drawerLayout.closeDrawer(drawerPanel)
            aboutPage.visibility == View.VISIBLE || settingsPage.visibility == View.VISIBLE -> navigateBack()
            else -> super.onBackPressed()
        }
    }

    private fun connectViews() {
        preview = findViewById(R.id.leaf_image)
        preview.clipToOutline = true
        emptyImageHint = findViewById(R.id.empty_image_hint)
        selectedFile = findViewById(R.id.selected_file)
        cropSelector = findViewById(R.id.crop_selector)
        analyzeButton = findViewById(R.id.analyze_image)
        progress = findViewById(R.id.progress)
        status = findViewById(R.id.status)
        resultPanel = findViewById(R.id.result_panel)
        resultCrop = findViewById(R.id.result_crop)
        resultDiagnosis = findViewById(R.id.result_diagnosis)
        resultConfidence = findViewById(R.id.result_confidence)
        resultPathogen = findViewById(R.id.result_pathogen)
        resultTreatment = findViewById(R.id.result_treatment)
        resultWarning = findViewById(R.id.result_warning)
    }

    private fun loadMetadata() {
        val text = assets.open("class_indices.json").bufferedReader().use { it.readText() }
        val root = JSONObject(text)
        val names = root.getJSONArray("class_names")
        classNames = List(names.length()) { names.getString(it) }
        classMetadata = root.getJSONObject("classes")
        require(classNames.size == 13) { "Expected 13 disease labels." }
        require(classMetadata?.length() == classNames.size) { "Disease labels and treatment data do not match." }
        require(classNames.all { classMetadata?.has(it) == true }) { "A disease label is missing treatment data." }
    }

    private fun loadModels() {
        diseaseInterpreter = loadInterpreter("disease_model.tflite")
        cropInterpreter = loadInterpreter("crop_type_model.tflite")
        val disease = diseaseInterpreter ?: error("The disease model is missing.")
        val crop = cropInterpreter ?: error("The crop model is missing.")
        require(disease.getInputTensor(0).shape().contentEquals(intArrayOf(1, imageSize, imageSize, 3))) {
            "The disease model input must be 128 by 128 RGB."
        }
        require(crop.getInputTensor(0).shape().contentEquals(intArrayOf(1, imageSize, imageSize, 3))) {
            "The crop model input must be 128 by 128 RGB."
        }
        require(disease.getOutputTensor(0).shape().last() == classNames.size) {
            "The disease model output does not match the label list."
        }
        require(crop.getOutputTensor(0).shape().last() == 2) {
            "The crop model must return Potato and Tomato scores."
        }
        require(disease.getInputTensor(0).dataType() == org.tensorflow.lite.DataType.FLOAT32) {
            "The disease model must use float image input."
        }
        require(crop.getInputTensor(0).dataType() == org.tensorflow.lite.DataType.FLOAT32) {
            "The crop model must use float image input."
        }
    }

    private fun loadInterpreter(fileName: String): Interpreter {
        val modelBytes = assets.open(fileName).use { it.readBytes() }
        val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
        modelBuffer.put(modelBytes)
        modelBuffer.rewind()
        val options = Interpreter.Options().setNumThreads(2)
        return Interpreter(modelBuffer, options)
    }

    private fun chooseImage() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "image/*"
        }
        startActivityForResult(intent, PICK_IMAGE_REQUEST)
    }

    private fun capturePhoto() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        val cameraDirectory = File(cacheDir, CAMERA_DIRECTORY)
        if (!cameraDirectory.exists() && !cameraDirectory.mkdirs()) {
            showStatus(getString(R.string.camera_storage_error))
            return
        }

        val photoFile = try {
            File.createTempFile("ipdds_", ".jpg", cameraDirectory)
        } catch (error: IOException) {
            showStatus("${getString(R.string.camera_storage_error)} ${error.message ?: ""}".trim())
            return
        }
        val photoUri = FileProvider.getUriForFile(this, "$packageName.fileprovider", photoFile)
        pendingCameraFile = photoFile
        intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
        intent.clipData = ClipData.newUri(contentResolver, "Leaf photo", photoUri)
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)

        try {
            startActivityForResult(intent, CAMERA_REQUEST)
        } catch (error: ActivityNotFoundException) {
            pendingCameraFile = null
            photoFile.delete()
            showStatus(getString(R.string.camera_unavailable))
        }
    }

    @Deprecated("Used for broad device compatibility with the system document picker.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode) {
            PICK_IMAGE_REQUEST -> {
                if (resultCode != RESULT_OK) return
                val uri = data?.data ?: run {
                    showStatus("The photo picker did not return an image.")
                    return
                }
                val hasReadPermission = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION != 0
                if (hasReadPermission && data.flags and Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION != 0) {
                    try {
                        contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    } catch (error: SecurityException) {
                        showStatus("Could not retain access to this photo: ${error.message ?: "permission denied"}")
                        return
                    }
                }
                showSelectedImage(uri)
            }

            CAMERA_REQUEST -> {
                val photoFile = pendingCameraFile
                pendingCameraFile = null
                if (resultCode != RESULT_OK) {
                    photoFile?.delete()
                    return
                }
                if (photoFile?.isFile != true || photoFile.length() == 0L) {
                    photoFile?.delete()
                    showStatus(getString(R.string.camera_photo_error))
                    return
                }
                val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", photoFile)
                showSelectedImage(uri, photoFile.name, photoFile)
            }

            else -> return
        }
    }

    private fun showSelectedImage(uri: Uri, displayName: String? = null, cameraFile: File? = null) {
        val previousCameraFile = selectedCameraFile
        if (previousCameraFile != null && previousCameraFile != cameraFile) {
            previousCameraFile.delete()
        }
        selectedCameraFile = cameraFile
        selectedImage = uri
        selectedFile.text = displayName
            ?: uri.lastPathSegment
                ?.substringAfterLast('/')
                ?.let(Uri::decode)
            ?: getString(R.string.no_image_selected)
        preview.setImageDrawable(null)
        emptyImageHint.visibility = View.VISIBLE
        showStatus("")
        resultPanel.visibility = View.GONE
        setBusy(true)
        worker.execute {
            try {
                val bitmap = decodeImage(uri, PREVIEW_MAX_SIZE)
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    preview.setImageBitmap(bitmap)
                    emptyImageHint.visibility = View.GONE
                    setBusy(false)
                }
            } catch (error: Exception) {
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    setBusy(false)
                    showStatus("Could not open this photo: ${error.message ?: "unsupported image"}")
                }
            }
        }
    }

    private fun analyzeImage() {
        val uri = selectedImage
        if (uri == null) {
            showStatus(getString(R.string.pick_photo_first))
            return
        }
        if (!modelReady) {
            showStatus("The models are not ready. Close and reopen the app.")
            return
        }

        resultPanel.visibility = View.GONE
        showStatus("")
        setBusy(true)
        val selectedCrop = cropSelector.selectedItemPosition
        worker.execute {
            try {
                val bitmap = decodeImage(uri, PREVIEW_MAX_SIZE)
                val result = try {
                    diagnose(bitmap, selectedCrop)
                } finally {
                    bitmap.recycle()
                }
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    showResult(result)
                    setBusy(false)
                }
            } catch (error: Exception) {
                runOnUiThread {
                    if (isFinishing || isDestroyed) return@runOnUiThread
                    setBusy(false)
                    showStatus(
                        if (error is NotLeafPhotoException) {
                            getString(R.string.leaf_photo_required)
                        } else {
                            "Could not analyze this photo: ${error.message ?: "unknown error"}"
                        }
                    )
                }
            }
        }
    }

    private fun diagnose(bitmap: Bitmap, selectedCrop: Int): Diagnosis {
        val input = bitmapToInput(bitmap)
        val cropOutput = Array(1) { FloatArray(2) }
        cropInterpreter?.run(input.duplicate().apply { rewind() }, cropOutput)
            ?: error("The crop model is not loaded.")
        require(cropOutput[0].all { it.isFinite() && it in 0f..1f }) {
            "The crop model returned invalid scores."
        }

        val detectedCropIndex = if (cropOutput[0][1] >= TOMATO_THRESHOLD) 1 else 0
        val cropConfidenceValue = cropOutput[0][detectedCropIndex]
        if (selectedCrop == 0 && cropConfidenceValue < MIN_CROP_CONFIDENCE) {
            throw UncertainLeafClassificationException()
        }
        val detectedCrop = if (detectedCropIndex == 0) "Potato" else "Tomato"
        val cropName = when (selectedCrop) {
            1 -> "Potato"
            2 -> "Tomato"
            else -> detectedCrop
        }
        val cropConfidence = if (selectedCrop == 0) {
            cropOutput[0][detectedCropIndex]
        } else {
            null
        }

        val diseaseOutput = Array(1) { FloatArray(classNames.size) }
        diseaseInterpreter?.run(input.duplicate().apply { rewind() }, diseaseOutput)
            ?: error("The disease model is not loaded.")
        require(diseaseOutput[0].all { it.isFinite() && it in 0f..1f }) {
            "The disease model returned invalid scores."
        }
        val allowedIndices = classNames.indices.filter { classNames[it].startsWith(cropName) }
        require(allowedIndices.isNotEmpty()) { "No disease categories were found for $cropName." }

        val allowedProbability = allowedIndices.sumOf { diseaseOutput[0][it].toDouble() }.toFloat()
        require(allowedProbability > 0f) { "The disease model returned invalid scores." }
        val bestIndex = allowedIndices.maxBy { diseaseOutput[0][it] }
        val confidence = diseaseOutput[0][bestIndex] / allowedProbability
        if (confidence < MIN_DISEASE_CONFIDENCE) {
            throw UncertainLeafClassificationException()
        }
        val rawName = classNames[bestIndex]
        val details = classMetadata?.getJSONObject(rawName)
            ?: error("Treatment details are missing for $rawName.")
        val recommendation = details.optString("recommendation")
        require(recommendation.isNotBlank()) { "Treatment details are missing for $rawName." }

        return Diagnosis(
            crop = cropName,
            cropConfidence = cropConfidence,
            disease = details.getString("display_name"),
            confidence = confidence,
            pathogen = details.optString("pathogen", "Not specified"),
            treatment = recommendation,
            cropIsUncertain = selectedCrop == 0 && cropConfidence != null && cropConfidence < CROP_WARNING_THRESHOLD
        )
    }

    private fun bitmapToInput(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
        val pixels = IntArray(imageSize * imageSize)
        resized.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)
        val hsv = FloatArray(3)
        var leafColorPixels = 0
        var centerLeafColorPixels = 0
        var leafBrightnessSum = 0.0
        var leafBrightnessSquaredSum = 0.0
        val centerPixels = imageSize * imageSize / 4
        val input = ByteBuffer.allocateDirect(imageSize * imageSize * 3 * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        pixels.forEachIndexed { index, color ->
            Color.colorToHSV(color, hsv)
            val isLeafColor = hsv[0] in MIN_LEAF_HUE..MAX_LEAF_HUE &&
                hsv[1] >= MIN_LEAF_SATURATION &&
                hsv[2] >= MIN_LEAF_BRIGHTNESS
            if (isLeafColor) {
                leafColorPixels++
                leafBrightnessSum += hsv[2]
                leafBrightnessSquaredSum += hsv[2] * hsv[2]
                val x = index % imageSize
                val y = index / imageSize
                if (x in imageSize / 4 until imageSize * 3 / 4 &&
                    y in imageSize / 4 until imageSize * 3 / 4
                ) {
                    centerLeafColorPixels++
                }
            }
            input.putFloat(((color shr 16) and 0xff).toFloat())
            input.putFloat(((color shr 8) and 0xff).toFloat())
            input.putFloat((color and 0xff).toFloat())
        }
        input.rewind()
        if (resized !== bitmap) resized.recycle()
        val meanLeafBrightness = leafBrightnessSum / max(leafColorPixels, 1)
        val leafBrightnessVariance =
            leafBrightnessSquaredSum / max(leafColorPixels, 1) - meanLeafBrightness * meanLeafBrightness
        val leafColorRatio = leafColorPixels.toFloat() / pixels.size
        if (leafColorRatio < MIN_LEAF_COLOR_RATIO ||
            leafColorRatio > MAX_LEAF_COLOR_RATIO ||
            centerLeafColorPixels.toFloat() / centerPixels < MIN_CENTER_LEAF_COLOR_RATIO ||
            leafBrightnessVariance < MIN_LEAF_BRIGHTNESS_VARIANCE
        ) {
            throw NotLeafPhotoException()
        }
        return input
    }

    private fun decodeImage(uri: Uri, maxDimension: Int): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        val boundsStream = contentResolver.openInputStream(uri)
            ?: throw FileNotFoundException("The selected photo could not be opened.")
        boundsStream.use { BitmapFactory.decodeStream(it, null, bounds) }
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "The selected file is not a supported image." }

        val sampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight, maxDimension)
        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val imageStream = contentResolver.openInputStream(uri)
            ?: throw FileNotFoundException("The selected photo could not be opened.")
        val bitmap = imageStream.use { BitmapFactory.decodeStream(it, null, options) }
            ?: throw FileNotFoundException("The selected photo could not be decoded.")
        return applyOrientation(uri, bitmap)
    }

    private fun calculateSampleSize(width: Int, height: Int, maxDimension: Int): Int {
        var sample = 1
        while (max(width / sample, height / sample) > maxDimension) {
            sample *= 2
        }
        return sample
    }

    private fun applyOrientation(uri: Uri, bitmap: Bitmap): Bitmap {
        val orientation = contentResolver.openInputStream(uri)?.use {
            ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } ?: ExifInterface.ORIENTATION_NORMAL
        val degrees = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (degrees == 0f) return bitmap
        val matrix = Matrix().apply { postRotate(degrees) }
        val oriented = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (oriented !== bitmap) bitmap.recycle()
        return oriented
    }

    private fun showResult(result: Diagnosis) {
        resultCrop.text = if (result.cropConfidence == null) {
            getString(R.string.result_crop_selected, result.crop)
        } else {
            getString(R.string.result_crop_confidence, result.crop, percent(result.cropConfidence))
        }
        resultDiagnosis.text = result.disease
        resultConfidence.text = getString(R.string.result_disease_confidence, percent(result.confidence))
        resultPathogen.text = result.pathogen
        resultTreatment.text = result.treatment
        resultWarning.visibility = if (result.cropIsUncertain) View.VISIBLE else View.GONE
        if (result.cropIsUncertain) resultWarning.text = getString(R.string.low_confidence)
        resultPanel.visibility = View.VISIBLE
        status.visibility = View.GONE
    }

    private fun percent(value: Float): String = String.format(Locale.getDefault(), "%.1f%%", value * 100f)

    private fun showStatus(message: String) {
        status.text = message
        status.visibility = if (message.isBlank()) View.GONE else View.VISIBLE
    }

    private fun setBusy(busy: Boolean) {
        progress.visibility = if (busy) View.VISIBLE else View.GONE
        analyzeButton.isEnabled = !busy && modelReady
        findViewById<Button>(R.id.choose_image).isEnabled = !busy
        capturePhotoButton.isEnabled = !busy
        capturePhotoButton.alpha = if (busy) 0.55f else 1f
    }

    override fun onDestroy() {
        val diseaseModel = diseaseInterpreter
        val cropModel = cropInterpreter
        diseaseInterpreter = null
        cropInterpreter = null
        worker.execute {
            diseaseModel?.close()
            cropModel?.close()
        }
        worker.shutdown()
        pendingCameraFile?.delete()
        pendingCameraFile = null
        selectedCameraFile?.delete()
        selectedCameraFile = null
        super.onDestroy()
    }

    private data class Diagnosis(
        val crop: String,
        val cropConfidence: Float?,
        val disease: String,
        val confidence: Float,
        val pathogen: String,
        val treatment: String,
        val cropIsUncertain: Boolean
    )

    private class NotLeafPhotoException : Exception()

    private class UncertainLeafClassificationException : Exception(
        "Could not confidently identify this leaf. Try a clear, close-up photo of one potato or tomato leaf."
    )

    companion object {
        private const val PICK_IMAGE_REQUEST = 1001
        private const val CAMERA_REQUEST = 1002
        private const val CAMERA_DIRECTORY = "camera"
        private const val PREVIEW_MAX_SIZE = 1024
        private const val TOMATO_THRESHOLD = 0.7f
        private const val CROP_WARNING_THRESHOLD = 0.7f
        private const val MIN_LEAF_HUE = 35f
        private const val MAX_LEAF_HUE = 155f
        private const val MIN_LEAF_SATURATION = 0.14f
        private const val MIN_LEAF_BRIGHTNESS = 0.12f
        private const val MIN_LEAF_COLOR_RATIO = 0.12f
        private const val MAX_LEAF_COLOR_RATIO = 0.85f
        private const val MIN_CENTER_LEAF_COLOR_RATIO = 0.08f
        private const val MIN_LEAF_BRIGHTNESS_VARIANCE = 0.0081
        private const val MIN_CROP_CONFIDENCE = 0.25f
        private const val MIN_DISEASE_CONFIDENCE = 0.35f
        private const val SETTINGS_PREFERENCES = "ipdds_settings"
        private const val KEY_REMEMBER_PLANT_SELECTION = "remember_plant_selection"
        private const val KEY_LAST_PLANT_SELECTION = "last_plant_selection"
    }
}
