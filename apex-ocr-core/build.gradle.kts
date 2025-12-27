// ApexOCR Core Module - Tensor Operations and Neural Network Engine

plugins {
    java
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

repositories {
    mavenCentral()
}

dependencies {
    // Core module has no internal dependencies
    
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
