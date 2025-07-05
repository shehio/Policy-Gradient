output "instance_id" {
  description = "Instance ID of the GPU EC2 instance"
  value       = aws_instance.gpu_runner.id
}

output "public_ip" {
  description = "Public IP address of the GPU EC2 instance"
  value       = aws_instance.gpu_runner.public_ip
}

output "vpc_id" {
  description = "VPC ID of the created VPC"
  value       = aws_vpc.main.id
}

output "subnet_id" {
  description = "Subnet ID of the created subnet"
  value       = aws_subnet.main.id
} 