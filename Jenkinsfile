pipeline {
    agent any

    stages {
        stage("SCM Checkout") {
            steps {
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
            }
        }
    }
}