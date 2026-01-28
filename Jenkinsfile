pipeline{
    any agent

    environment{
        IMAGE_NAME='bindusdet/docker-app:${GIT_COMMIT}'
    }

    stages{
        stage('git checkout'){
            steps{
                git url: 'https://github.com/bindusdet/docker.git', branch: 'app-1'
            }
        }
        stage('build-stage'){
            steps{
                sh'''
                printenv 
                docker build -t $IMAGE_NAME .
                '''
            }
        }
        stage('testing stage'){
            steps{
                sh'''
                docker run -it -d --name chatbot -p 8501:8501 ${IMAGE_NAME}
                '''
            }
        }
    }

}