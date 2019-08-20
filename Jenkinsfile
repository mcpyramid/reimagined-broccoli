node {
    ansiColor('xterm') {
        stage("SCM") {
            checkout([
                $class: 'GitSCM', 
                branches: [[name: '*/master']], 
                doGenerateSubmoduleConfigurations: false, 
                extensions: [], 
                submoduleCfg: [], 
                userRemoteConfigs: [                        
                    credentialsId: 'f2adcce0-abee-4e43-8a9b-fd1001dbdbf7', 
                    url: 'https://github.com/mcpyramid/reimagined-broccoli.git'
                ]
            ])
        },
        stage("Build and Learn") {
            docker.image('docker.io/python:3.6.9').inside {
                echo "Installing wget and sagemaker"
                pip3 install wget
                pip3 install sagemaker
            }
        },
        stage("Evaluate Model") {
            //
        },
        stage("QA") {
            //
        }
        
    }
}