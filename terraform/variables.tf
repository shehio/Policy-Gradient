variable "aws_region" {
  description = "AWS region to deploy resources in"
  default     = "us-east-1"
}

variable "ami_id" {
  description = "AMI ID for Deep Learning AMI (Ubuntu) with GPU support"
  default     = "ami-05ee60afff9d0a480"
}

variable "instance_type" {
  description = "EC2 instance type with GPU support"
  default     = "g4dn.xlarge"
}

variable "key_name" {
  description = "Name of the AWS EC2 Key Pair to use for SSH access"
  default     = "generic"
} 