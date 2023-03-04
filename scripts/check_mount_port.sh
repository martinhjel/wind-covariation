resourceGroupName="rg-basic"
storageAccountName="stamartihj"

# This command assumes you have logged in with az login
httpEndpoint=$(az storage account show \
    --resource-group $resourceGroupName \
    --name $storageAccountName \
    --query "primaryEndpoints.file" --output tsv | tr -d '"')
smbPath=$(echo $httpEndpoint | cut -c7-${#httpEndpoint})
fileHost=$(echo $smbPath | tr -d "/")

nc -zvw3 $fileHost 445