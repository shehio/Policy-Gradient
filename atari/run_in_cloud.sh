#!/bin/bash
## Assuming you have a key pair called in the current region and generic.pem in the current directory
cd terraform
terraform init
terraform plan
terraform apply -auto-approve
ssh -i generic.pem ubuntu@<instance-public-ip>
