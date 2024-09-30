import click
import subprocess
import os
from string import Template

PLATFORM_CONFIGS = {
    "aws": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
    "gcp": ["GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT_ID"],
    "azure": [
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
    ],
}


@click.group()
@click.version_option()
def cli():
    "A CLI tool for multinode infrastructure"


@cli.command(name="login")
@click.argument("platform", type=click.Choice(list(PLATFORM_CONFIGS.keys())))
def login(platform):
    "Login to a cloud platform"
    "Login to a cloud platform"
    if platform == "aws":
        click.echo("Running aws configure...")
        subprocess.run(["aws", "configure"], check=True)
        click.echo(
            "AWS configuration complete. You can verify your settings with 'aws configure list'."
        )
    else:
        click.echo("Platform not supported")


@cli.command(name="show")
@click.argument("item", type=click.Choice(["capacity", "subnets"]))
def show(item):
    "Show the current capacity and subnets"
    if item == "capacity":
        click.echo("Running aws ec2 describe-capacity-reservations...")
        subprocess.run(["aws", "ec2", "describe-capacity-reservations"], check=True)
    elif item == "subnets":
        click.echo("Running aws ec2 describe-subnets...")
        subprocess.run(["aws", "ec2", "describe-subnets"], check=True)
    else:
        click.echo("Item not supported")


@cli.command(name="configure")
@click.option(
    "--security-key",
    prompt="Enter your security key",
    help="The security key for the cluster, which is usually the NAME of the .pem file",
)
@click.option(
    "--public-subnet-id",
    prompt="Enter the public subnet ID",
    help="You can check with 'sila show subnets'",
)
@click.option(
    "--private-subnet-id",
    prompt="Enter the private subnet ID",
    help="You can check with 'sila show subets'",
)
@click.option(
    "--capacity-reservation-id",
    prompt="Enter the capacity reservation ID",
    help="You can check with 'sila show capacity'",
)
@click.option(
    "--compute-node-type",
    prompt="The type of compute nodes that the capacity reservation is for\n0: A100s (40gb) -> p4d.24xlarge\n1: A100s (80gb) -> p4de.24xlarge\n2: H100s (80gb) -> p5d.48xlarge\n3: H200s (144gb) -> p5de.48xlarge\nEnter your compute node type:",
    help="The type of compute nodes that the capacity reservation is for\n0: A100s (40gb) -> p4d.24xlarge\n1: A100s (80gb) -> p4de.24xlarge\n2: H100s (80gb) -> p5d.48xlarge\n3: H200s (144gb) -> p5de.48xlarge",
)
@click.option(
    "--number-of-compute-nodes",
    prompt="Enter the number of compute nodes",
    type=int,
    help="The number of compute nodes that the capacity reservation is for",
)
@click.option(
    "--output",
    default=".",
    help="Output directory for the YAML file. Default is the current directory.",
)
@click.argument("item", type=click.Choice(["cluster", "subnet"]))
def configure(
    security_key,
    public_subnet_id,
    private_subnet_id,
    capacity_reservation_id,
    compute_node_type,
    number_of_compute_nodes,
    item,
    output,
):
    "Configure the cluster and subnet"
    if item == "cluster":
        cluster_info = {
            "security_key": security_key,
            "public_subnet_id": public_subnet_id,
            "private_subnet_id": private_subnet_id,
            "capacity_reservation_id": capacity_reservation_id,
            "compute_node_type": compute_node_type,
            "number_of_compute_nodes": number_of_compute_nodes,
        }
        if compute_node_type == "0":
            cluster_info["compute_node_type"] = "p4d.24xlarge"
        elif compute_node_type == "1":
            cluster_info["compute_node_type"] = "p4de.24xlarge"
        elif compute_node_type == "2":
            cluster_info["compute_node_type"] = "p5d.48xlarge"
        elif compute_node_type == "3":
            cluster_info["compute_node_type"] = "p5de.48xlarge"
        else:
            click.echo("Invalid compute node type")
            return

        with open("template.yaml", "r") as file:
            template_content = file.read()

        template = Template(template_content)
        result = template.substitute(cluster_info)

        # Save the result to a new YAML file in the specified directory
        output_file = os.path.join(output, "config.yaml")
        with open(output_file, "w") as file:
            file.write(result)

    elif item == "subnet":
        click.echo("Enter information to configure the subnet")
    else:
        click.echo("Item not supported")


@cli.command(name="launch")
@click.argument("item", type=click.Choice(["cluster", "subnet"]))
@click.option(
    "-n",
    "--cluster-name",
    prompt="Enter the name of the cluster",
    help="The name of the cluster to launch",
)
def deploy(item, cluster_name):
    "Deploy the cluster and subnet"
    if item == "cluster":
        click.echo("Running sila deploy cluster...")
        if cluster_name is None:
            cluster_name = "cluster"
        subprocess.run(
            ["pcluster", "create-cluster", "config.yaml", "-n", cluster_name],
            check=True,
        )
    elif item == "subnet":
        click.echo("Running sila deploy subnet...")
        subprocess.run(["sila", "deploy", "subnet"], check=True)
    else:
        click.echo("Item not supported")
