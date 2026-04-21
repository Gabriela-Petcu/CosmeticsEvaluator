using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace CosmeticsEvaluator.Api.Migrations
{
    /// <inheritdoc />
    public partial class InitialCreate : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
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
                    PricePerOunce = table.Column<double>(type: "REAL", nullable: true),
                    CategoryAntiAging = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryAcneTreatments = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryExfoliators = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryEyeTreatments = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryFaceMasks = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryFaceOils = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryFaceSerums = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryFaceSunscreen = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryFaceWash = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryFacialPeels = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryMistsEssences = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryMoisturizerTreatments = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryMoisturizers = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryNightCreams = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryToners = table.Column<int>(type: "INTEGER", nullable: false),
                    CategoryBlottingPapers = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ProductCatalog", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "Users",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    SkinType = table.Column<string>(type: "TEXT", nullable: false),
                    MainConcern = table.Column<string>(type: "TEXT", nullable: false),
                    BudgetLevel = table.Column<string>(type: "TEXT", nullable: false),
                    Email = table.Column<string>(type: "TEXT", nullable: false),
                    PasswordHash = table.Column<string>(type: "TEXT", nullable: false),
                    Role = table.Column<string>(type: "TEXT", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Users", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "EvaluationHistory",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Brand = table.Column<string>(type: "TEXT", nullable: false),
                    Name = table.Column<string>(type: "TEXT", nullable: false),
                    Price = table.Column<double>(type: "REAL", nullable: false),
                    NOfReviews = table.Column<int>(type: "INTEGER", nullable: false),
                    NOfLoves = table.Column<int>(type: "INTEGER", nullable: false),
                    PricePerOunce = table.Column<double>(type: "REAL", nullable: false),
                    ProductId = table.Column<string>(type: "TEXT", nullable: false),
                    ReviewScore = table.Column<double>(type: "REAL", nullable: false),
                    MlProbability = table.Column<double>(type: "REAL", nullable: false),
                    FinalVerdict = table.Column<string>(type: "TEXT", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "TEXT", nullable: false),
                    UserId = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_EvaluationHistory", x => x.Id);
                    table.ForeignKey(
                        name: "FK_EvaluationHistory_Users_UserId",
                        column: x => x.UserId,
                        principalTable: "Users",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_EvaluationHistory_UserId",
                table: "EvaluationHistory",
                column: "UserId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "EvaluationHistory");

            migrationBuilder.DropTable(
                name: "ProductCatalog");

            migrationBuilder.DropTable(
                name: "Users");
        }
    }
}
