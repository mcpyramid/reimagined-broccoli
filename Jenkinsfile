node {
    checkout scm
    /*
     * In order to communicate with the MySQL server, this Pipeline explicitly
     * maps the port (`3306`) to a known port on the host machine.
     */
    docker.image('python:3.6.9').inside {
        sh 'echo "Installing wget and sagemaker"'
        sh 'pip3 install wget'
        sh 'pip3 install sagemaker'
    }
}

// pipeline {
//     agent any

//     stages {
//         stage("SCM Checkout") {
//             steps {
//                 checkout([
//                     $class: 'GitSCM', 
//                     branches: [[name: '*/master']], 
//                     doGenerateSubmoduleConfigurations: false, 
//                     extensions: [], 
//                     submoduleCfg: [], 
//                     userRemoteConfigs: [                        
//                         credentialsId: 'f2adcce0-abee-4e43-8a9b-fd1001dbdbf7', 
//                         url: 'https://github.com/mcpyramid/reimagined-broccoli.git'
//                     ]
//                 ])  
//             }
//         }
//         stage("Build and Learn") {
//             steps {               
//                 docker.image('docker.io/python:3.6.9').inside {
//                     echo "Installing wget and sagemaker"
//                     pip3 install wget
//                     pip3 install sagemaker
//                 }
//             }
//         }
//     }
// }