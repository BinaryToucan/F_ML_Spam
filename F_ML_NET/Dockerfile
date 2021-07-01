FROM mcr.microsoft.com/dotnet/runtime:5.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:5.0 AS build
WORKDIR /src
COPY ["F_ML_NET/F_ML_NET.fsproj", "F_ML_NET/"]
RUN dotnet restore "F_ML_NET/F_ML_NET.fsproj"
COPY . .
WORKDIR "/src/F_ML_NET"
RUN dotnet build "F_ML_NET.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "F_ML_NET.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "F_ML_NET.dll"]
