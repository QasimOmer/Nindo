package com.coderobust.mlandroidapp

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.io.FileInputStream



class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var ageInput: EditText
    private lateinit var weightInput: EditText
    private lateinit var heightInput: EditText
    private lateinit var resultText: TextView
    private lateinit var buttonPredict: Button
    private lateinit var progressBar: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ageInput = findViewById(R.id.ageInput)
        weightInput = findViewById(R.id.weightInput)
        heightInput = findViewById(R.id.heightInput)
        resultText = findViewById(R.id.resultText)
        buttonPredict = findViewById(R.id.predictButton)
        progressBar = findViewById(R.id.riskProgressBar)

        // Load TensorFlow Lite model
        interpreter = loadModel("risk_model.tflite")

        // Predict on button click
        buttonPredict.setOnClickListener {
            val age = ageInput.text.toString().toFloatOrNull() ?: 0.0f
            val weight = weightInput.text.toString().toFloatOrNull() ?: 0.0f
            val height = heightInput.text.toString().toFloatOrNull() ?: 0.0f

            val riskScore = predictRisk(age, weight, height)
            resultText.text = if (riskScore > 0.5) {
                "High Risk (Score: $riskScore)"
            } else {
                "Low Risk (Score: $riskScore)"
            }
            progressBar.progress = (riskScore * 100).toInt()
            Log.d("MainActivity", "Risk Score: $riskScore")
            Log.d("MainActivity", "Progress: ${progressBar.progress}")
            Log.d("MainActivity", "Result Text: ${resultText.text}")
        }
    }

    // Function to load the model from the assets folder
    private fun loadModel(modelPath: String): Interpreter {
        val assetFileDescriptor = assets.openFd(modelPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        return Interpreter(buffer)
    }

    // Function to predict the risk based on inputs (age, weight, height)
    private fun predictRisk(age: Float, weight: Float, height: Float): Float {
        // Manually scale the inputs (standardize them as done during training)
        val scaledAge = scaleInput(age, mean = 5.0f, std = 2.0f) // Example values for mean and std
        val scaledWeight = scaleInput(weight, mean = 4.5f, std = 0.7f)
        val scaledHeight = scaleInput(height, mean = 65.0f, std = 5.0f)

        // Prepare the input buffer (3 features)
        val inputBuffer = ByteBuffer.allocateDirect(3 * 4).order(ByteOrder.nativeOrder())
        inputBuffer.putFloat(scaledAge)
        inputBuffer.putFloat(scaledWeight)
        inputBuffer.putFloat(scaledHeight)

        // Prepare the output buffer (1 value: risk score)
        val outputBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())

        // Run the model
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        return outputBuffer.float
    }

    // Function to scale inputs manually
    private fun scaleInput(value: Float, mean: Float, std: Float): Float {
        return (value - mean) / std
    }
}
