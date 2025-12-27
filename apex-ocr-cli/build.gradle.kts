// ApexOCR CLI Module - Command Line Interface

plugins {
    application
}

application {
    mainClass.set("com.apexocr.cli.DemoMain")
    applicationName = "apex-ocr-demo"
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":apex-ocr-engine"))

    // CLI dependencies
    implementation("info.picocli:picocli:4.7.5")

    // Testing - JUnit 4
    testImplementation("junit:junit:4.13.2")
}

tasks.withType<JavaCompile> {
    options.encoding = "UTF-8"
    options.compilerArgs.addAll(listOf(
        "-Xlint:unchecked",
        "-Xlint:deprecation"
    ))
}

tasks.named<Test>("test") {
    useJUnit()
    maxHeapSize = "2g"
}
