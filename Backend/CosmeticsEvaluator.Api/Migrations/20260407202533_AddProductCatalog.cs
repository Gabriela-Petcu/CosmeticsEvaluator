using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CosmeticsEvaluator.Api.Migrations
{
    /// <inheritdoc />
    public partial class AddProductCatalog : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "Brand",
                table: "EvaluationHistory",
                type: "TEXT",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<int>(
                name: "NOfLoves",
                table: "EvaluationHistory",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<int>(
                name: "NOfReviews",
                table: "EvaluationHistory",
                type: "INTEGER",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "Name",
                table: "EvaluationHistory",
                type: "TEXT",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<double>(
                name: "Price",
                table: "EvaluationHistory",
                type: "REAL",
                nullable: false,
                defaultValue: 0.0);

            migrationBuilder.AddColumn<double>(
                name: "PricePerOunce",
                table: "EvaluationHistory",
                type: "REAL",
                nullable: false,
                defaultValue: 0.0);

            migrationBuilder.CreateTable(
                name: "ProductCatalog",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Brand = table.Column<string>(type: "TEXT", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    Price = table.Column<double>(type: "REAL", nullable: false),
                    NOfReviews = table.Column<int>(type: "INTEGER", nullable: false),
                    NOfLoves = table.Column<int>(type: "INTEGER", nullable: false),
                    ReviewScore = table.Column<double>(type: "REAL", nullable: false),
                    PricePerOunce = table.Column<double>(type: "REAL", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ProductCatalog", x => x.Id);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "ProductCatalog");

            migrationBuilder.DropColumn(
                name: "Brand",
                table: "EvaluationHistory");

            migrationBuilder.DropColumn(
                name: "NOfLoves",
                table: "EvaluationHistory");

            migrationBuilder.DropColumn(
                name: "NOfReviews",
                table: "EvaluationHistory");

            migrationBuilder.DropColumn(
                name: "Name",
                table: "EvaluationHistory");

            migrationBuilder.DropColumn(
                name: "Price",
                table: "EvaluationHistory");

            migrationBuilder.DropColumn(
                name: "PricePerOunce",
                table: "EvaluationHistory");
        }
    }
}
