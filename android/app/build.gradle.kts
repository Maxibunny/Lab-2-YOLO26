plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.example.furnituredetector"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.example.furnituredetector"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }
    buildTypes { release { isMinifyEnabled = false } }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += "tflite" }
}
dependencies {
    // LiteRT 1.x bundles the compatible Interpreter CPU API under org.tensorflow.lite.
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("androidx.documentfile:documentfile:1.0.1")
}
