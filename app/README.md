Запуск:

docker build -t ghostbim21/diagnosis-app:v3 .
docker login
docker push ghostbim21/diagnosis-app:v3

helm upgrade --install diagnosis-app ./chart
helm upgrade diagnosis-app ./chart

Использовалось для мониторинга:
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install monitoring prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace

kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80