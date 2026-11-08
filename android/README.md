# IPDDS for Android

IPDDS 1.1 means Intelligent Plant Disease Detection System. The app checks Potato and Tomato leaves offline using the embedded crop and 13-class disease models, with matching treatment information. The shared models use Potato and Tomato images from both configured PlantVillage sources. AI results are estimates and should be confirmed before treatment.

## Open and build

Open the `android` folder in Android Studio, allow Gradle sync to finish, then choose **Build > Build APK(s)**.

The debug APK is written to:

`android/app/build/outputs/apk/debug/app-debug.apk`

You can install it on a connected Android phone with:

```powershell
.\gradlew.bat installDebug
```

Or install the APK directly:

```powershell
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

The project targets Android 7.0 or newer. Choose a saved image with the system photo picker or take a new one with the phone's camera app. It does not request photo-library, camera, or internet permissions.

The interface follows the device's system light or dark appearance. Use the sidebar to open the leaf checker, About, and Settings. About shows the two on-device models and their validation details. Its launcher icon uses a leaf mark on a dark background.

## How image checks work

Choose or take a photo and tap **Check leaf**. IPDDS estimates whether the crop is Potato or Tomato and shows that estimate and its confidence with the disease result. A close-up is checked twice, once on the whole leaf and once on the center. A wider plant photo is checked as a grid of leaf-sized areas; if those areas disagree, the strongest reading is shown with a warning. Low-confidence crop estimates are rejected. A local color check rejects blank or visibly unrelated images before a diagnosis. This is a heuristic, not a separately trained leaf detector, so important results need confirmation. The selected image is converted to RGB, letterboxed to the model size recorded in `class_indices.json` (160 by 160 for the current models), and passed to both TensorFlow Lite models as float values from 0 to 255. The model's own rescaling layer performs the same normalization used during training.

The crop estimate uses the larger of the two model probabilities; estimates below 70% are not used to produce a disease result. Disease names and treatments are loaded from the same class metadata used by the web app.

The app runs inference locally. Its result is a guide and should not replace professional diagnosis.
