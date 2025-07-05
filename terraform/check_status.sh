#!/bin/bash

# Get the instance ID from Terraform output
echo "Getting instance ID from Terraform..."
INSTANCE_ID=$(terraform output -raw instance_id 2>/dev/null || echo "")

if [ -z "$INSTANCE_ID" ]; then
    echo "Error: Could not get instance ID from Terraform output"
    echo "Make sure you've run 'terraform apply' and the instance is created"
    exit 1
fi

echo "Instance ID: $INSTANCE_ID"

# Check instance status
echo "Checking instance status..."
aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text

# Get the public IP
echo "Getting public IP..."
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Public IP: $PUBLIC_IP"

if [ "$PUBLIC_IP" != "None" ] && [ -n "$PUBLIC_IP" ]; then
    echo ""
    echo "To SSH into the instance:"
    echo "ssh -i your-key.pem ubuntu@$PUBLIC_IP"
    echo ""
    echo "To check the user data logs:"
    echo "ssh -i your-key.pem ubuntu@$PUBLIC_IP 'sudo cat /var/log/user-data.log'"
    echo ""
    echo "To check if the training is running:"
    echo "ssh -i your-key.pem ubuntu@$PUBLIC_IP 'ps aux | grep python'"
    echo ""
    echo "To check the training log:"
    echo "ssh -i your-key.pem ubuntu@$PUBLIC_IP 'tail -f /home/ubuntu/Policy-Gradient/training.log'"
else
    echo "Instance doesn't have a public IP yet. It might still be starting up."
fi 