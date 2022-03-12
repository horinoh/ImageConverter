// ImageConverter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#pragma warning(push)
#pragma warning(disable:4819)
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#pragma warning(pop)
#ifdef _DEBUG
#pragma comment(lib, "opencv_world454d.lib")
#else
#pragma comment(lib, "opencv_world454.lib")
#endif

#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <string_view>
#include <charconv>
#include <numeric>

namespace CV
{
	static void Preview(std::string_view Title, const cv::Mat Image)
	{
		cv::imshow(data(Title), Image);
		cv::waitKey(0);
		cv::destroyWindow(data(Title));
	}
	static void Preview(std::string_view Title, const cv::Mat& Image, const cv::Size& Size)
	{
		cv::Mat Resized;
		cv::resize(Image, Resized, Size, 0, 0, cv::INTER_NEAREST);
		Preview(Title, Resized);
	}
	static void DrawGrid(const cv::Mat& Image, const cv::Size& GridSize)
	{
		const auto Color = cv::Scalar(0, 0, 0);
		const auto SizeY = Image.size().height / GridSize.height;
		for (auto i = 0; i < SizeY; ++i) {
			cv::line(Image, cv::Point(0, i * GridSize.height), cv::Point(Image.size().width, i * GridSize.height), Color);
		}
		const auto SizeX = Image.size().width / GridSize.width;
		for (auto i = 0; i < SizeX; ++i) {
			cv::line(Image, cv::Point(i * GridSize.width, 0), cv::Point(i * GridSize.width, Image.size().height), Color);
		}
	}
	static void ColorReduction(cv::Mat& Dst, const cv::Mat& Image, const uint32_t ColorCount)
	{
		//!< 1行の行列となるように変形
		cv::Mat Points;
		Image.convertTo(Points, CV_32FC3);
		Points = Points.reshape(3, Image.rows * Image.cols);

		//!< k-means クラスタリング
		cv::Mat_<int> Clusters(Points.size(), CV_32SC1);
		cv::Mat Centers;
		cv::kmeans(Points, ColorCount, Clusters, cv::TermCriteria(cv::TermCriteria::Type::EPS | cv::TermCriteria::Type::MAX_ITER, 10, 1.0), 1, cv::KmeansFlags::KMEANS_PP_CENTERS, Centers);

		//!< 各ピクセル値を属するクラスタの中心値で置き換え
		Dst = cv::Mat(Image.size(), Image.type());
		auto It = Dst.begin<cv::Vec3b>();
		for (auto i = 0; It != Dst.end<cv::Vec3b>(); ++It, ++i) {
			const auto Color = Centers.at<cv::Vec3f>(Clusters(i), 0);
			(*It)[0] = cv::saturate_cast<uchar>(Color[0]); //!< B
			(*It)[1] = cv::saturate_cast<uchar>(Color[1]); //!< G
			(*It)[2] = cv::saturate_cast<uchar>(Color[2]); //!< R
		}
	}
	static void GrayScale(cv::Mat& Dst, const cv::Mat& Image)
	{
		cvtColor(Image, Dst, cv::COLOR_BGR2GRAY);

		//!< 二値化
		//cv::threshold(Dst, Dst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); //!< 大津アルゴリズムを用いて最適な閾値を決定する
	}
}

template<uint8_t W, uint8_t H>
class Converter
{
public:
	Converter(const cv::Mat& Img) : Image(Img) {}

	class MapEntity
	{
	public:
		uint32_t PatternIndex = 0;
		uint32_t Flags = 0;
	};

	using Palette = std::vector<uint32_t>;

	//!< 32 ビットカラーパターン or カラーインデックスパターンを持たせられる
	using PatternEntity = std::array<std::array<uint32_t, W>, H>;
	//!< パレットインデックス + カラーインデックスパターン
	class Pattern
	{
	public:
		bool HasValidPaletteIndex() const { return 0xffffffff != PaletteIndex; }
		uint32_t PaletteIndex = 0xffffffff;
		PatternEntity ColorIndices;
	};

	virtual uint16_t ToPlatformColor(const cv::Vec3b& Color) const { return 0; }
	virtual cv::Vec3b FromPlatformColor(const uint16_t& Color) const { return cv::Vec3b(0, 0, 0); }

	virtual PatternEntity& ToPlatformColorPattern(PatternEntity& lhs, const cv::Mat& rhs) const {
		for (auto i = 0; i < rhs.rows; ++i) {
			for (auto j = 0; j < rhs.cols; ++j) {
				lhs[i][j] = ToPlatformColor(rhs.ptr<cv::Vec3b>(i)[j]);
			}
		}
		return lhs;
	}
	virtual Pattern& ToIndexColorPattern(Pattern& Pat, const uint32_t PalIdx, const PatternEntity& ColPat) {
		const auto& Pal = Palettes[(Pat.PaletteIndex = PalIdx)];
		for (auto i = 0; i < size(ColPat); ++i) {
			for (auto j = 0; j < size(ColPat[i]); ++j) {
				Pat.ColorIndices[i][j] = static_cast<uint32_t>(std::distance(begin(Pal), std::ranges::find(Pal, ColPat[i][j])));
			}
		}
		return Pat;
	}

	virtual uint16_t GetPaletteCount() const = 0;
	virtual uint16_t GetPaletteColorCount() const = 0;

	//!< パレットの先頭に予約色(透明色、背景色)を持つか
	virtual bool HasPaletteReservedColor() const { return true; }
	virtual uint16_t GetPaletteReservedColorCount() const { return HasPaletteReservedColor() ? 1 : 0; }
	virtual uint16_t GetPaletteReservedColor() const { return 0x0000; }

	virtual cv::Size GetMapSize(const uint8_t w, const uint8_t h) const { return cv::Size(Image.cols / w, Image.rows / h); }
	virtual cv::Size GetMapSize() const { return GetMapSize(W, H); }

#pragma region CREATE
	virtual Converter& Create() {
		CreateMap();
		CreatePalette();
		CreatePattern();
		return *this;
	}

	virtual Converter& CreateMap() {
		const auto MapSize = GetMapSize();
		for (auto i = 0; i < MapSize.height; ++i) {
			auto& MapEnt = Map.emplace_back();
			for (auto j = 0; j < MapSize.width; ++j) {
				const cv::Mat cvPat = Image(cv::Rect(j * W, i * H, W, H));
#if 0
				//!< 反転して既存になるものは追加しない #TODO
				cv::Mat cvPatV, cvPatH, cvPatVH;
				cv::flip(cvPat, cvPatV, 0);
				cv::flip(cvPat, cvPatH, 1);
				cv::flip(cvPat, cvPatVH, -1);

				PatternEntity Pat, PatV, PatH, PatVH;
				ToPlatformColorPattern(Pat, cvPat);
				ToPlatformColorPattern(PatV, cvPatV);
				ToPlatformColorPattern(PatH, cvPatH);
				ToPlatformColorPattern(PatVH, cvPatVH);
				const auto It = std::ranges::find_if(ColorPatterns, [&](const PatternEntity& rhs) { return rhs == Pat || rhs == PatV || rhs == PatH || rhs == PatVH; });
#else
				PatternEntity Pat;
				ToPlatformColorPattern(Pat, cvPat);
				const auto It = std::ranges::find_if(ColorPatterns, [&](const PatternEntity& rhs) { return rhs == Pat; });
#endif
				if (end(ColorPatterns) != It) {
					MapEnt.emplace_back(MapEntity({ .PatternIndex = static_cast<uint32_t>(std::distance(begin(ColorPatterns), It)), .Flags = 0 }));
				}
				else {
					MapEnt.emplace_back(MapEntity({ .PatternIndex = static_cast<uint32_t>(size(ColorPatterns)), .Flags = 0 }));
					ColorPatterns.emplace_back(Pat);
				}
			}
		}
		return *this;
	}

	void AddPatternColorToPalette(Palette& Pal, const PatternEntity& Pat)
	{
		for (auto i : Pat) {
			for (auto j : i) {
				if (end(Pal) == std::ranges::find(Pal, j)) {
					Pal.emplace_back(j);
				}
			}
		}
	}
	//!< パターン毎に 1 パレットとするケース
	void CreatePalettePerPattern() {
		Palettes.clear();
		for (const auto& p : ColorPatterns) {
			auto& Pal = Palettes.emplace_back();
			AddPatternColorToPalette(Pal, p);

			std::ranges::sort(Pal);
		}
	}
	//!< マップの列毎に 1 パレットとするケース
	void CreatePalettePerMapRow() {
		Palettes.clear();
		for (const auto& r : Map) {
			auto& Pal = Palettes.emplace_back();

			//!< 列は同じパレットを使わなければならない
			for (const auto& c : r) {
				AddPatternColorToPalette(Pal, ColorPatterns[c.PatternIndex]);
			}

			std::ranges::sort(Pal);
		}
	}
	//!< マップの 2 x 2 毎に 1 パレットとするケース
	void CreatePalettePerMap2x2() {
		Palettes.clear();
		for (auto i = 0; i < size(Map); i += 2) {
			for (auto j = 0; j < size(Map[i]); j += 2) {
				auto& Pal = Palettes.emplace_back();

				//!< 2 x 2 部分は同じパレットを使わなければならない
				AddPatternColorToPalette(Pal, ColorPatterns[Map[i + 0][j + 0].PatternIndex]);
				AddPatternColorToPalette(Pal, ColorPatterns[Map[i + 0][j + 1].PatternIndex]);
				AddPatternColorToPalette(Pal, ColorPatterns[Map[i + 1][j + 0].PatternIndex]);
				AddPatternColorToPalette(Pal, ColorPatterns[Map[i + 1][j + 1].PatternIndex]);

				std::ranges::sort(Pal);
			}
		}
	}
#if 1
	virtual Converter& CreatePalette() { CreatePalettePerPattern(); return *this; }
#elif 0
	virtual Converter& CreatePalette() { CreatePalettePerMapRow(); return *this; }
#else
	virtual Converter& CreatePalette() { CreatePalettePerMap2x2(); return *this; }
#endif

	void CreatePattern(const std::vector<uint32_t>& PalInds) {
		Patterns.reserve(size(ColorPatterns));
		for (auto i = 0; i < size(ColorPatterns); ++i) {
			ToIndexColorPattern(Patterns.emplace_back(), PalInds[i], ColorPatterns[i]);
		}
	}
	void CreatePatternPerMapRow(const std::vector<uint32_t>& PalInds) {
		Patterns.resize(size(ColorPatterns));

		for (auto i = 0; i < size(Map); ++i) {
			const auto PalInd = PalInds[PalInds[i]];
			for (auto j : Map[i]) {
				if (!Patterns[j.PatternIndex].HasValidPaletteIndex()) {
					ToIndexColorPattern(Patterns[j.PatternIndex], PalInd, ColorPatterns[j.PatternIndex]);
				}
			}
		}
	}
	void CreatePatternPerMap2x2(const std::vector<uint32_t>& PalInds) {
		Patterns.resize(size(ColorPatterns));

		auto k = 0;
		for (auto i = 0; i < size(Map); i += 2) {
			for (auto j = 0; j < size(Map[i]); j += 2) {
				const auto PalInd = PalInds[PalInds[k++]];

				auto PatInd = Map[i + 0][j + 0].PatternIndex;
				if (!Patterns[PatInd].HasValidPaletteIndex()) {
					ToIndexColorPattern(Patterns[PatInd], PalInd, ColorPatterns[PatInd]);
				}
				PatInd = Map[i + 0][j + 1].PatternIndex;
				if (!Patterns[PatInd].HasValidPaletteIndex()) {
					ToIndexColorPattern(Patterns[PatInd], PalInd, ColorPatterns[PatInd]);
				}
				PatInd = Map[i + 1][j + 0].PatternIndex;
				if (!Patterns[PatInd].HasValidPaletteIndex()) {
					ToIndexColorPattern(Patterns[PatInd], PalInd, ColorPatterns[PatInd]);
				}
				PatInd = Map[i + 1][j + 1].PatternIndex;
				if (!Patterns[PatInd].HasValidPaletteIndex()) {
					ToIndexColorPattern(Patterns[PatInd], PalInd, ColorPatterns[PatInd]);
				}
			}
		}
	}
	virtual Converter& CreatePattern() {
		std::vector<uint32_t> PaletteIndices(size(Palettes));
		std::iota(begin(PaletteIndices), end(PaletteIndices), 0);

		//!< パレットをまとめる
		while ([&]() {
			for (auto i = 0; i < size(Palettes); ++i) {
				for (auto j = i + 1; j < size(Palettes); ++j) {
					auto& lhs = Palettes[i];
					auto& rhs = Palettes[j];
					if (!empty(lhs) && !empty(rhs)) {
						std::vector<uint32_t> Union;
						std::ranges::set_union(lhs, rhs, std::back_inserter(Union));
						//!< パレットの和集合が パレット中のカラー数以下に収まる場合は、一つのパレットにまとめる事が可能
						if (GetPaletteColorCount() - GetPaletteReservedColorCount() > size(Union)) {
							lhs.assign(begin(Union), end(Union));
							rhs.clear();
							std::ranges::replace(PaletteIndices, j, i);
							return true;
						}
					}
				}
			}
			return false;
			}()) {
		}

		//!< パレット番号を詰める
		{
			auto SortUnique = PaletteIndices;
			std::ranges::sort(SortUnique);
			const auto [B, E] = std::ranges::unique(SortUnique);
			SortUnique.erase(B, E);
			for (auto i = 0; i < size(SortUnique); ++i) {
				std::ranges::replace(PaletteIndices, SortUnique[i], i);
			}
		}
		//!< 空になったパレットは消す
		{
			const auto [B, E] = std::ranges::remove_if(Palettes, [](const std::vector<uint32_t>& rhs) { return empty(rhs); });
			Palettes.erase(B, E);
		}

		//!< インデックスカラーのパターンを作成
		Patterns.clear();
#if 1
		CreatePattern(PaletteIndices);
#elif 0
		CreatePatternPerMapRow(PaletteIndices);
#else
		CreatePatternPerMap2x2(PaletteIndices);
#endif
		return *this;
	}
#pragma endregion

#pragma region OUTPUT
	//!< 型を指定してのパレット出力
	template<typename T>
	void OutputPaletteOfType(std::string_view Name) const {
		std::cout << "\tPalette count = " << size(Palettes) << " / " << GetPaletteCount() << (size(Palettes) > GetPaletteCount() ? " warning" : "") << std::endl;

		std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
		assert(!OutBin.bad());
		std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
		assert(!OutText.bad());

		//OutText << "const " << typeid(T).name() << " " << Name << "[] = {" << std::endl;
		OutText << "const u" << (sizeof(T) << 3) << " " << Name << "[] = {" << std::endl;

		for (auto i : Palettes) {
			const auto MaxCount = GetPaletteColorCount() - GetPaletteReservedColorCount();
			std::cout << "\t\tPalette color count = " << size(i) << " / " << MaxCount << (size(i) > MaxCount ? " warning" : "" ) << std::endl;

			const T TransparentColor = 0; //!< 先頭色 (ここでは 0 としている)

			//!< 出力用の型へ変換
			std::vector<T> PalOut;
			{
				//!< 透明色
				if (HasPaletteReservedColor()) {
					PalOut.emplace_back(TransparentColor);
				}
				std::ranges::copy(i, std::back_inserter(PalOut));
				//!< 空きも透明色で埋める
				for (auto j = size(PalOut); j < GetPaletteColorCount(); j++) {
					PalOut.emplace_back(TransparentColor);
				}
			}

			//!< 出力
			OutText << "\t";
			for (auto j = 0; j < size(PalOut); ++j) {
				OutText << "0x" << std::hex << std::setw(sizeof(PalOut[j]) << 1) << std::right << std::setfill('0') << static_cast<uint16_t>(PalOut[j]);
				if (size(PalOut) - 1 > j) { OutText << ", "; }
			}
			OutText << std::endl;

			OutBin.write(reinterpret_cast<const char*>(data(PalOut)), size(PalOut) * sizeof(PalOut[0]));
		}
		OutText << "};" << std::endl;

		OutBin.close();
		OutText.close();
	}
	virtual const Converter& OutputPalette(std::string_view Name) const { return *this; }
	virtual const Converter& OutputPattern(std::string_view Name) const { return *this; }
	virtual const Converter& OutputMap(std::string_view Name) const {
		std::cout << "\tMap size = " << size(this->Map[0]) << " x " << size(this->Map) << std::endl;

		std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
		assert(!OutBin.bad());
		std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
		assert(!OutText.bad());

		//OutText << "const " << typeid(uint8_t).name() << " " << Name << "[] = {" << std::endl;
		OutText << "const u" << (sizeof(uint8_t) << 3) << " " << Name << "[] = {" << std::endl;

		for (auto i = 0; i < size(this->Map); ++i) {
			OutText << "\t";
			for (auto j = 0; j < size(this->Map[i]); ++j) {
				const auto PatIdx8 = static_cast<uint8_t>(this->Map[i][j].PatternIndex);

				OutText << "0x" << std::hex << std::setw(sizeof(PatIdx8) << 1) << std::right << std::setfill('0') << static_cast<uint16_t>(PatIdx8);
				if (size(this->Map) - 1 > i || size(this->Map[i]) - 1 > j) { OutText << ", "; }

				OutBin.write(reinterpret_cast<const char*>(&PatIdx8), sizeof(PatIdx8));
			}
			OutText << std::endl;
		}
		OutText << "};" << std::endl;

		OutBin.close();
		OutText.close();

		return *this;
	}
	virtual const Converter& OutputBAT(std::string_view Name) const { return *this; }
	virtual const Converter& OutputAnimation(std::string_view Path) const {
		std::cout << "\tSprite count = " << size(Map) << std::endl;
		std::cout << "\tMax animation count = " << size(Map[0]) << std::endl;
		for (const auto& r : Map) {
			std::cout << "\t\tSprite animations = ";
			for (const auto& c : r) {
				std::cout << c.PatternIndex << ", ";
			}
			std::cout << std::endl;
		}
		return *this;
	}
#pragma endregion

#pragma region RESTORE
	virtual const Converter& RestorePalette() const {
#ifdef _DEBUG
		const auto Count = GetPaletteColorCount() - GetPaletteReservedColorCount();
		cv::Mat Res(cv::Size(Count, static_cast<int>(size(Palettes))), Image.type());
		for (auto i = 0; i < size(Palettes); ++i) {
			for (auto j = 0; j < Count; ++j) {
				Res.ptr<cv::Vec3b>(i)[j] = j < size(Palettes[i]) ? FromPlatformColor(Palettes[i][j]) : cv::Vec3b();
			}
		}
		CV::Preview("Palette", Res, Res.size() * 50);
#endif
		return *this;
	}
	virtual const Converter& RestorePattern() const {
#ifdef _DEBUG
		constexpr auto ColumnCount = 16; //!< デバッグ表示する際のカラム数 (横に長くなるので適当な所で折り返す)
		const auto PatCount = static_cast<int>(size(Patterns));
		cv::Mat Res(cv::Size(ColumnCount * W, (PatCount / ColumnCount + 1) * H), Image.type());
		for (auto p = 0; p < size(Patterns); ++p) {
			const auto& Pat = Patterns[p];
			assert(Pat.HasValidPaletteIndex());
			const auto& Pal = Palettes[Pat.PaletteIndex];
			cv::Mat cvPat(cv::Size(W, H), Image.type());
			for (auto i = 0; i < H; ++i) {
				for (auto j = 0; j < W; ++j) {
					cvPat.ptr<cv::Vec3b>(i)[j] = FromPlatformColor(Pal[Pat.ColorIndices[i][j]]);
				}
			}
			cvPat.copyTo(Res(cv::Rect((p % ColumnCount) * W, (p / ColumnCount) * H, W, H)));
		}

		CV::DrawGrid(Res, cv::Size(W, H));
		CV::Preview("Pattern", Res, Res.size() * 5);
#endif
		return *this;
	}
	virtual const Converter& RestoreMap() const {
#ifdef _DEBUG
		cv::Mat Res(Image.size(), Image.type());

		for (auto r = 0; r < size(Map); ++r) {
			for (auto c = 0; c < size(Map[r]); ++c) {
				const auto& MapEnt = Map[r][c];

				const auto& Pat = Patterns[MapEnt.PatternIndex];
				assert(Pat.HasValidPaletteIndex());
				//MapEnt.Flags; //!< 反転情報等

				cv::Mat cvPat(cv::Size(W, H), Image.type());
				for (auto i = 0; i < size(Pat.ColorIndices); ++i) {
					for (auto j = 0; j < size(Pat.ColorIndices[i]); ++j) {
						cvPat.ptr<cv::Vec3b>(i)[j] = FromPlatformColor(Palettes[Pat.PaletteIndex][Pat.ColorIndices[i][j]]);
					}
				}
				cvPat.copyTo(Res(cv::Rect(c * W, r * H, W, H)));
			}
		}

		CV::DrawGrid(Res, cv::Size(W, H));
		CV::Preview("Map", Res, Res.size() * 3);
#endif
		return *this;
	}
#pragma endregion

protected:
	const cv::Mat& Image;
	std::vector<PatternEntity> ColorPatterns;

	std::vector<std::vector<MapEntity>> Map;
	std::vector<Palette> Palettes;
	std::vector<Pattern> Patterns;
};

class ResourceReaderBase
{
public:
	ResourceReaderBase() {
		//!< LOG_LEVEL_INFO だと gtk 周りの余計なログが出てうざいので
		cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
	}

	void Read(std::string_view Path) {
		std::filesystem::current_path(Path);
		for (const auto& i : std::filesystem::directory_iterator(std::filesystem::current_path())) {
			if (!i.is_directory()) {
				//!< .res ファイルを探す (Search for .res files)
				if (i.path().has_extension() && ".res" == i.path().extension().string()) {
					std::cout << std::filesystem::absolute(i.path()).string() << std::endl;
					std::ifstream In(std::filesystem::absolute(i.path()).string(), std::ios::in);
					if (!In.fail()) {
						//!< 行を読み込む (Read line)
						std::string Line;
						while (std::getline(In, Line)) {
							//!< 項目を読み込む (Read items)
							std::vector<std::string> Items;
							std::stringstream SS(Line);
							std::string Item;
							while (std::getline(SS, Item, ' ')) {
								Items.emplace_back(Item);
							}

							if (!empty(Items)) {
								//!< 画像ファイル名から " を取り除く (Remove " from image file name)
								std::erase(Items[2], '"');
								//const auto FilePath = std::filesystem::absolute(std::filesystem::path(Items[2])).string();
								const auto FilePath = Items[2];

								if ("PALETTE" == Items[0]) {
									ProcessPalette(Items[1], FilePath);
								}
								if ("TILESET" == Items[0]) {
									ProcessTileSet(Items[1], FilePath, size(Items) > 3 ? Items[3] : "", size(Items) > 4 ? Items[4] : "");
								}
								if ("ITILESET" == Items[0]) {
									ProcessImageTileSet(Items[1], FilePath, size(Items) > 3 ? Items[3] : "", size(Items) > 4 ? Items[4] : "");
								}
								if ("MAP" == Items[0]) {
									uint32_t MapBase = 0;
									if (size(Items) > 5) {
										auto [ptr, ec] = std::from_chars(data(Items[5]), data(Items[5]) + size(Items[5]), MapBase);
										if (std::errc() != ec) {}
									}
									ProcessMap(Items[1], FilePath, Items[3], size(Items) > 4 ? Items[4] : "", MapBase);
								}
								if ("IMAP" == Items[0]) {
									uint32_t MapBase = 0;
									if (size(Items) > 5) {
										auto [ptr, ec] = std::from_chars(data(Items[5]), data(Items[5]) + size(Items[5]), MapBase);
										if (std::errc() != ec) {}
									}
									ProcessImageMap(Items[1], FilePath, Items[3], size(Items) > 4 ? Items[4] : "", MapBase);
								}
								if ("SPRITE" == Items[0]) {
									uint32_t Width = 0;
									auto [ptr0, ec0] = std::from_chars(data(Items[3]), data(Items[3]) + size(Items[3]), Width);
									if (std::errc() != ec0) {}

									uint32_t Height = 0;
									auto [ptr1, ec1] = std::from_chars(data(Items[4]), data(Items[4]) + size(Items[4]), Height);
									if (std::errc() != ec1) {}

									uint32_t Time = 0;
									if (size(Items) > 6) {
										auto [ptr, ec] = std::from_chars(data(Items[6]), data(Items[6]) + size(Items[6]), Time);
										if (std::errc() != ec) {}
									}

									uint32_t Iteration = 500000;
									if (size(Items) > 9) {
										auto [ptr, ec] = std::from_chars(data(Items[9]), data(Items[9]) + size(Items[9]), Iteration);
										if (std::errc() != ec) {}
									}

									ProcessSprite(Items[1], FilePath, Width, Height, size(Items) > 5 ? Items[5] : "", Time, size(Items) > 7 ? Items[7] : "", size(Items) > 8 ? Items[8] : "", Iteration);
								}
							}
						}
						In.close();
					}
				}
			}
		}
	}
	virtual void ProcessPalette(std::string_view Name, std::string_view File) {}
	virtual void ProcessTileSet(std::string_view Name, std::string_view File, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] std::string_view Option) {}
	virtual void ProcessImageTileSet(std::string_view Name, std::string_view File, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] std::string_view Option) {}
	virtual void ProcessMap(std::string_view Name, std::string_view File, std::string_view TileSet, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Mapbase) {}
	virtual void ProcessImageMap(std::string_view Name, std::string_view File, std::string_view TileSet, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Mapbase) {}
	virtual void ProcessSprite(std::string_view Name, std::string_view File, const uint32_t Width, const uint32_t Height, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Time, [[maybe_unused]] std::string_view Collision, [[maybe_unused]] std::string_view Option, [[maybe_unused]] const uint32_t Iteration) {}

	virtual void Clear(std::string_view Name) {
		std::filesystem::remove(std::string(Name) + ".bin");
		std::filesystem::remove(std::string(Name) + ".text");
	}
	virtual void ClearPalette(std::string_view Name) { Clear(Name); }
	virtual void ClearTileSet(std::string_view Name) { Clear(Name); }
	virtual void ClearMap(std::string_view Name) { Clear(Name); }
	virtual void ClearSprite(std::string_view Name) { Clear(Name); }

};

#pragma region PCE
/*
* 画面		: 256 x 224
* BG		: 512 x 256(6 種類ある)、(15 + 1) 色 x 16 パレット (9 ビットカラー = 8 x 8 x 8 = 512 色)
* スプライト	: 64 枚、16 x 16 等(6 種類ある)、(15 + 1) 色 x 16 パレット (9 ビットカラー)
*/
namespace PCE
{
	template<uint8_t W, uint8_t H>
	class ConverterBase : public Converter<W, H>
	{
	private:
		using Super = Converter<W, H>;
	public:
		ConverterBase(const cv::Mat& Img) : Super(Img) {}

		virtual uint16_t ToPlatformColor(const cv::Vec3b& Color) const override { return ((Color[1] >> 5) << 6) | ((Color[2] >> 5) << 3) | (Color[0] >> 5); }
		virtual cv::Vec3b FromPlatformColor(const uint16_t& Color) const override { return cv::Vec3b((Color & 0x7) << 5, ((Color & (0x7 << 6)) >> 6) << 5, ((Color & (0x7 << 3)) >> 3) << 5); }

		virtual uint16_t GetPaletteCount() const override { return 16; };
		virtual uint16_t GetPaletteColorCount() const override { return 16; }

		virtual const ConverterBase& OutputPalette(std::string_view Name) const override {
			this->OutputPaletteOfType<uint16_t>(Name);
			return *this;
		}
		virtual uint8_t PaletteIndexShift() const { return 0; };
		virtual const ConverterBase& OutputPatternPalette(std::string_view Name) const {
			std::ofstream OutBin(data(std::string(Name) + ".pal" + ".bin"), std::ios::binary | std::ios::out);
			assert(!OutBin.bad());
			std::ofstream OutText(data(std::string(Name) + ".pal" + ".txt"), std::ios::out);
			assert(!OutText.bad());

			//OutText << "const " << typeid(uint8_t).name() << " " << Name << "_PAL[] = {" << std::endl;
			OutText << "const u" << (sizeof(uint8_t) << 3) << " " << Name << "_PAL[] = {" << std::endl;

			for (auto i = 0; i < size(this->Patterns); ++i) {
				const auto& Pat = this->Patterns[i];

				//!< パターン毎のパレットインデックス (BG では 4 ビットシフトする必要がある)
				assert(Pat.HasValidPaletteIndex());
				const uint8_t PalIdx = Pat.PaletteIndex << PaletteIndexShift();

				OutText << "\t0x" << std::hex << std::setw(sizeof(PalIdx) << 1) << std::right << std::setfill('0') << static_cast<uint16_t>(PalIdx);
				if (size(this->Patterns) - 1 > i) { OutText << ", "; }
				OutText << std::endl;

				OutBin.write(reinterpret_cast<const char*>(&PalIdx), sizeof(PalIdx));
			}
			OutText << "};" << std::endl;

			OutBin.close();
			OutText.close();

			return *this;
		}
	};

	namespace Image
	{
		//!< イメージ : スクロールさせない静止画向き
		//!< 4 プレーンに分けて出力、4 プレーンを合わせるとカラーインデックスが求まる
		//!< パターン 8 x 8 を表すのに
		//!<	最初の u16 x 8 の 上位下位 8 ビットへプレーン 0, 1、続く u16 x 8 の 上位下位 8 ビットへプレーン 2, 3
		//!<	u16[00] 1111111100000000
		//!<	u16[01] 1111111100000000
		//!<	....
		//!<	u16[14] 3333333322222222
		//!<	u16[15] 3333333322222222
		//!<  
		//!<	Background Attribute Table (BAT)
		//!<	BAT を用いて、使用パレットはパターン毎に指定が可能 
		//!<    LLLLTTTT TTTTTTTT
		//!<	L : パレット番号[0, 15]、T : パターン番号 [0, 4095](内ユーザが使えるのは[256, 4095])
		template<uint8_t W = 8, uint8_t H = 8>
		class Converter : public ConverterBase<W, H>
		{
		private:
			using Super = ConverterBase<W, H>;
		public:
			Converter(const cv::Mat& Img) : Super(Img) {}

			virtual Converter& Create() override { Super::Create(); return *this; }

			virtual const Converter& OutputPattern(std::string_view Name) const override {
				std::cout << "\tPattern count = " << size(this->Patterns) << std::endl;

				std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
				assert(!OutBin.bad());
				std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
				assert(!OutText.bad());

				//OutText << "const " << typeid(uint16_t).name() << " " << Name << "[] = {" << std::endl;
				OutText << "const u" << (sizeof(uint16_t) << 3) << " " << Name << "[] = {" << std::endl;

				for (auto pat = 0; pat < size(this->Patterns); ++pat) {
					const auto& Pat = this->Patterns[pat];
					//!< 2 プレーン
					for (auto pl = 0; pl < 2; ++pl) {
						OutText << "\t";
						for (auto i = 0; i < size(Pat.ColorIndices); ++i) {
							uint16_t Plane = 0;
							for (auto j = 0; j < size(Pat.ColorIndices[i]); ++j) {
								const auto ColorIndex = Pat.ColorIndices[i][j] + this->GetPaletteReservedColorCount(); //!< 先頭の透明色を考慮
								const auto ShiftL = 7 - j;
								const auto ShiftU = ShiftL + 8;
								const auto MaskL = 1 << ((pl << 1) + 0);
								const auto MaskU = 1 << ((pl << 1) + 1);
								Plane |= ((ColorIndex & MaskL) ? 1 : 0) << ShiftL;
								Plane |= ((ColorIndex & MaskU) ? 1 : 0) << ShiftU;
							}
							OutText << "0x" << std::hex << std::setw(sizeof(Plane) << 1) << std::right << std::setfill('0') << Plane;
							if (size(this->Patterns) - 1 > pat || 1 > pl || size(Pat.ColorIndices) - 1 > i) { OutText << ", "; }

							OutBin.write(reinterpret_cast<const char*>(&Plane), sizeof(Plane));
						}
					}
					OutText << std::endl;
				}
				OutText << "};" << std::endl;

				OutBin.close();
				OutText.close();

				return *this;
			}
			//!< BAT はパターン番号とパレット番号からなるマップ
			virtual const Converter& OutputBAT(std::string_view Name) const override {
				std::cout << "\tBAT size = " << size(this->Map[0]) << " x " << size(this->Map) << std::endl;

				std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
				assert(!OutBin.bad());
				std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
				assert(!OutText.bad());

				//OutText << "const " << typeid(uint16_t).name() << " " << Name << "[] = {" << std::endl;
				OutText << "const u" << (sizeof(uint16_t) << 3) << " " << Name << "[] = {" << std::endl;

				for (auto i = 0; i < size(this->Map); ++i) {
					OutText << "\t";
					for (auto j = 0; j < size(this->Map[i]); ++j) {
						const auto PatIdx = this->Map[i][j].PatternIndex;
						assert(this->Patterns[PatIdx].HasValidPaletteIndex());

						//!< アプリから使用できるパターンインデックスは 256 以降 [256, 4095] なのでオフセット
						const uint16_t BAT = (this->Patterns[PatIdx].PaletteIndex << 12) | (PatIdx + 256);
						OutBin.write(reinterpret_cast<const char*>(&BAT), sizeof(BAT));

						OutText << "0x" << std::hex << std::setw(sizeof(BAT) << 1) << std::right << std::setfill('0') << BAT;
						if (size(this->Map) - 1 > i || size(this->Map[i]) - 1 > j) { OutText << ", "; }
					}
					OutText << std::endl;
				}
				OutText << "};" << std::endl;

				OutBin.close();
				OutText.close();

				return *this;
			}
		};
	}

	namespace BG
	{
		//!< BG : スクロールさせる背景
		//!< 4 プレーンに分けて出力、4 プレーンを合わせるとカラーインデックスが求まる
		//!< タイル 16 x 16 を並べてマップを作成することになる (8 x 8 ではなく 16 x 16 単位なので注意) 
		//!< タイルの 1 / 4 部分である 8 x 8 を表すのに以下のようになる
		//!<	最初の u16 x 8 の 上位下位 8 ビットへプレーン 0, 1、続く u16 x 8 の 上位下位 8 ビットへプレーン 2, 3
		//!<	u16[00] 1111111100000000
		//!<	u16[01] 1111111100000000
		//!<	....
		//!<	u16[14] 3333333322222222
		//!<	u16[15] 3333333322222222
		//!<	
		//!<	タイル 16 x 16 を表すには 4 回繰り返す、LT, RT, LB, RB の順
		//!<  
		//!< 使用パレットはタイル毎に指定が可能、タイル数分だけパレット番号を別途出力する(4 ビットシフトする必要がある)
		//!< 
		//!< マップは単純にタイル番号の uint8_t 羅列となる
		template<uint32_t W = 16, uint32_t H = 16>
		class Converter : public ConverterBase<W, H>
		{
		private:
			using Super = ConverterBase<W, H>;
		public:
			Converter(const cv::Mat& Img) : Super(Img) {}

			virtual Converter& Create() override { Super::Create(); return *this; }

			virtual const Converter& OutputPattern(std::string_view Name) const override {
				std::cout << "\tPattern count = " << size(this->Patterns) << std::endl;

				std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
				assert(!OutBin.bad());
				std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
				assert(!OutText.bad());

				//OutText << "const " << typeid(uint16_t).name() << " " << Name << "[] = {" << std::endl;
				OutText << "const u" << (sizeof(uint16_t) << 3) << " " << Name << "[] = {" << std::endl;

				//!< 16 x 16 のパターンを 4 つの 8 x 8 部分 (LT, RT, LB, RB) に分けて出力する
				for (auto pat = 0; pat < size(this->Patterns); ++pat) {
					const auto& Pat = this->Patterns[pat];

					const auto h = size(Pat.ColorIndices) >> 1;
					const auto w = size(Pat.ColorIndices[0]) >> 1;

					//!< LT (左上 8 x 8)
					//!< 2 プレーン
					for (auto pl = 0; pl < 2; ++pl) {
						OutText << "\t";
						//!< 8 x 8 部分
						for (auto i = 0; i < h; ++i) {
							uint16_t Plane = 0;
							for (auto j = 0; j < w; ++j) {
								const auto ColorIndex = Pat.ColorIndices[i][j] + this->GetPaletteReservedColorCount(); //!< 先頭の透明色を考慮
								const auto ShiftL = 7 - j;
								const auto ShiftU = ShiftL + 8;
								const auto MaskL = 1 << ((pl << 1) + 0);
								const auto MaskU = 1 << ((pl << 1) + 1);
								Plane |= ((ColorIndex & MaskL) ? 1 : 0) << ShiftL;
								Plane |= ((ColorIndex & MaskU) ? 1 : 0) << ShiftU;
							}
							OutText << "0x" << std::hex << std::setw(sizeof(Plane) << 1) << std::right << std::setfill('0') << Plane << ", ";

							OutBin.write(reinterpret_cast<const char*>(&Plane), sizeof(Plane));
						}
					}
					OutText << std::endl;

					//!< RT (右上 8 x 8)
					for (auto pl = 0; pl < 2; ++pl) {
						OutText << "\t";
						for (auto i = 0; i < h; ++i) {
							uint16_t Plane = 0;
							for (auto j = 0; j < w; ++j) {
								const auto ColorIndex = Pat.ColorIndices[i][j + w] + this->GetPaletteReservedColorCount();
								const auto ShiftL = 7 - j;
								const auto ShiftU = ShiftL + 8;
								const auto MaskL = 1 << ((pl << 1) + 0);
								const auto MaskU = 1 << ((pl << 1) + 1);
								Plane |= ((ColorIndex & MaskL) ? 1 : 0) << ShiftL;
								Plane |= ((ColorIndex & MaskU) ? 1 : 0) << ShiftU;
							}
							OutText << "0x" << std::hex << std::setw(sizeof(Plane) << 1) << std::right << std::setfill('0') << Plane << ", ";

							OutBin.write(reinterpret_cast<const char*>(&Plane), sizeof(Plane));
						}
					}
					OutText << std::endl;

					//!< LB (左下 8 x 8)
					for (auto pl = 0; pl < 2; ++pl) {
						OutText << "\t";
						for (auto i = 0; i < h; ++i) {
							uint16_t Plane = 0;
							for (auto j = 0; j < w; ++j) {
								const auto ColorIndex = Pat.ColorIndices[i + h][j] + this->GetPaletteReservedColorCount();
								const auto ShiftL = 7 - j;
								const auto ShiftU = ShiftL + 8;
								const auto MaskL = 1 << ((pl << 1) + 0);
								const auto MaskU = 1 << ((pl << 1) + 1);
								Plane |= ((ColorIndex & MaskL) ? 1 : 0) << ShiftL;
								Plane |= ((ColorIndex & MaskU) ? 1 : 0) << ShiftU;
							}
							OutText << "0x" << std::hex << std::setw(sizeof(Plane) << 1) << std::right << std::setfill('0') << Plane << ", ";

							OutBin.write(reinterpret_cast<const char*>(&Plane), sizeof(Plane));
						}
					}
					OutText << std::endl;

					//!< RB (右下 8 x 8)
					for (auto pl = 0; pl < 2; ++pl) {
						OutText << "\t";
						for (auto i = 0; i < h; ++i) {
							uint16_t Plane = 0;
							for (auto j = 0; j < w; ++j) {
								const auto ColorIndex = Pat.ColorIndices[i + h][j + w] + this->GetPaletteReservedColorCount();
								const auto ShiftL = 7 - j;
								const auto ShiftU = ShiftL + 8;
								const auto MaskL = 1 << ((pl << 1) + 0);
								const auto MaskU = 1 << ((pl << 1) + 1);
								Plane |= ((ColorIndex & MaskL) ? 1 : 0) << ShiftL;
								Plane |= ((ColorIndex & MaskU) ? 1 : 0) << ShiftU;
							}
							OutText << "0x" << std::hex << std::setw(sizeof(Plane) << 1) << std::right << std::setfill('0') << Plane;
							if (size(this->Patterns) - 1 > pat || 1 > pl || h - 1 > i) { OutText << ", "; }

							OutBin.write(reinterpret_cast<const char*>(&Plane), sizeof(Plane));
						}
					}
					OutText << std::endl;
				}
				OutText << "};" << std::endl;

				OutBin.close();
				OutText.close();

				return *this;
			}
			virtual uint8_t PaletteIndexShift() const override { return 4; };
		};
	}

	namespace Sprite {
		//!< スプライト
		//!< 4 プレーンに分けて出力、4 プレーンを合わせるとカラーインデックスが求まる
		template<uint8_t W = 16, uint8_t H = 16>
		class Converter : public ConverterBase<W, H>
		{
		private:
			using Super = ConverterBase<W, H>;
		public:
			Converter(const cv::Mat& Img) : Super(Img) {}

			virtual Converter& Create() override { Super::Create(); return *this; }

			//virtual Converter& CreatePalette() { this->CreatePalettePerMapRow(); return *this; }

			virtual const Converter& OutputPattern(std::string_view Name) const override {
				std::cout << "\tPattern count = " << size(this->Patterns) << std::endl;
				std::cout << "\tSprite size = " << static_cast<uint16_t>(W) << " x " << static_cast<uint16_t>(H) << std::endl;

				std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
				assert(!OutBin.bad());
				std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
				assert(!OutText.bad());

				//OutText << "const " << typeid(uint16_t).name() << " " << Name << "[] = {" << std::endl;
				OutText << "const u" << (sizeof(uint16_t) << 3) << " " << Name << "[] = {" << std::endl;

				for (auto pat = 0; pat < size(this->Patterns); ++pat) {
					const auto& Pat = this->Patterns[pat];
					//!< パターン毎のパレットインデックス情報を出力
					assert(Pat.HasValidPaletteIndex());
					std::cout << "\t\tPalette index = " << Pat.PaletteIndex << std::endl;

					//!< 4 プレーン
					for (auto pl = 0; pl < 4; ++pl) {
						OutText << "\t";
						for (auto i = 0; i < size(Pat.ColorIndices); ++i) {
							uint16_t Plane = 0;
							for (auto j = 0; j < size(Pat.ColorIndices[i]); ++j) {
								const auto ColorIndex = Pat.ColorIndices[i][j] + this->GetPaletteReservedColorCount(); //!< 先頭の透明色を考慮
								const auto Shift = 15 - j;
								const auto Mask = 1 << pl;
								Plane |= ((ColorIndex & Mask) ? 1 : 0) << Shift;
							}
							OutText << "0x" << std::hex << std::setw(sizeof(Plane) << 1) << std::right << std::setfill('0') << Plane;
							if (size(this->Patterns) - 1 > pat || 3 > pl || size(Pat.ColorIndices) - 1 > i) { OutText << ", "; }

							OutBin.write(reinterpret_cast<const char*>(&Plane), sizeof(Plane));
						}
					}
					OutText << std::endl;
				}
				OutText << "};" << std::endl;

				OutBin.close();
				OutText.close();

				return *this;
			}
		};
	}

	class ResourceReader : public ResourceReaderBase
	{
	private:
		using Super = ResourceReaderBase;
	public:
		virtual void ProcessPalette(std::string_view Name, std::string_view File) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Palette ] " << Name << " (" << File << ")" << std::endl;
#if 0
				Image::Converter<>(Image).Create().OutputPalette(Name).RestorePalette();
#else
				BG::Converter<>(Image).Create().OutputPalette(Name).RestorePalette();
#endif
			}
		}
		virtual void ProcessTileSet(std::string_view Name, std::string_view File, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] std::string_view Option) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Pattern ] " << Name << " (" << File << ")" << std::endl;
				BG::Converter<>(Image).Create().OutputPattern(Name).OutputPatternPalette(Name).RestorePattern();
			}
		}
		virtual void ProcessImageTileSet(std::string_view Name, std::string_view File, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] std::string_view Option) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Pattern ] " << Name << " (" << File << ")" << std::endl;
				//!< イメージの場合はパターンが全部異なったりするので、マップ(BAT) を復元するのと大して変わらない
				Image::Converter<>(Image).Create().OutputPattern(Name);
			}
		}
		virtual void ProcessMap(std::string_view Name, std::string_view File, std::string_view TileSet, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Mapbase) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Map ] " << Name << " (" << File << ")" << std::endl;
				BG::Converter<>(Image).Create().OutputMap(Name).RestoreMap();
			}
		}
		virtual void ProcessImageMap(std::string_view Name, std::string_view File, std::string_view TileSet, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Mapbase) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output BAT ] " << Name << " (" << File << ")" << std::endl;
				Image::Converter<>(Image).Create().OutputBAT(Name).RestoreMap();
			}
		}
		virtual void ProcessSprite(std::string_view Name, std::string_view File, const uint32_t Width, const uint32_t Height, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Time, [[maybe_unused]] std::string_view Collision, [[maybe_unused]] std::string_view Option, [[maybe_unused]] const uint32_t Iteration) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Sprite ] " << Name << " (" << File << ")" << std::endl;

				//!< 16x16, 16x32, 16x64, 32x16, 32x32, 32x64, 
				switch (Width << 3) {
				case 16:
					switch (Height << 3) {
					case 16:
						Sprite::Converter<16, 16>(Image).Create().OutputPattern(Name).OutputPatternPalette(Name).OutputAnimation(Name).RestorePattern();
						break;
					case 32:
						Sprite::Converter<16, 32>(Image).Create().OutputPattern(Name).OutputPatternPalette(Name).OutputAnimation(Name).RestorePattern();
						break;
					case 64:
						Sprite::Converter<16, 64>(Image).Create().OutputPattern(Name).OutputPatternPalette(Name).OutputAnimation(Name).RestorePattern();
						break;
					default:
						std::cerr << "Sprite size not supported" << std::endl;
						break;
					}
					break;
				case 32:
					switch (Height << 3) {
					case 16:
						Sprite::Converter<32, 16>(Image).Create().OutputPattern(Name).OutputPatternPalette(Name).OutputAnimation(Name).RestorePattern();
						break;
					case 32:
						Sprite::Converter<32, 32>(Image).Create().OutputPattern(Name).OutputPatternPalette(Name).OutputAnimation(Name).RestorePattern();
						break;
					case 64:
						Sprite::Converter<32, 64>(Image).Create().OutputPattern(Name).OutputPatternPalette(Name).OutputAnimation(Name).RestorePattern();
						break;
					default:
						std::cerr << "Sprite size not supported" << std::endl;
						break;
					}
					break;
				default: 
					std::cerr << "Sprite size not supported" << std::endl;
					break;
				}
			}
		}
		virtual void ClearTileSet(std::string_view Name) override {
			Super::ClearTileSet(Name);
			std::filesystem::remove(std::string(Name) + ".pal" + ".bin");
			std::filesystem::remove(std::string(Name) + ".pal" + ".text");
		}
	};
}
#pragma endregion //!< PCE

#pragma region FC
/*
* 画面		: 256 x 240
* BG		: 256 x 240(32 x 30 セル) x 2 画面、パターン 8 x 8 x 256 種、(3 + 1) 色 x 4 パレット (52色中)
* スプライト	: 8x8、64 枚、パターン 8 x 8 x 256 種、(3 + 1) 色 x 4 パレット (52色中)
*/
namespace FC {
#define TO_BGR(r, g, b) cv::Vec3b(b, g, r)
	//!< FC では固定の 64 色分のエントリがある (ただし同色もあるので実質は 52 色)
	static const std::array ColorEntries = {
		TO_BGR(117, 117, 117),
		TO_BGR(39,  27, 143),
		TO_BGR(0,   0, 171),
		TO_BGR(71,   0, 159),
		TO_BGR(143,   0, 119),
		TO_BGR(171,   0,  19),
		TO_BGR(167,   0,   0),
		TO_BGR(127,  11,   0),
		TO_BGR(67,  47,   0),
		TO_BGR(0,  71,   0),
		TO_BGR(0,  81,   0),
		TO_BGR(0,  63,  23),
		TO_BGR(27,  63,  95),
		TO_BGR(0,   0,   0),
		TO_BGR(0,   0,   0),
		TO_BGR(0,   0,   0),
		TO_BGR(188, 188, 188),
		TO_BGR(0, 115, 239),
		TO_BGR(35,  59, 239),
		TO_BGR(131,   0, 243),
		TO_BGR(191,   0, 191),
		TO_BGR(231,   0,  91),
		TO_BGR(219,  43,   0),
		TO_BGR(203,  79,  15),
		TO_BGR(139, 115,   0),
		TO_BGR(0, 151,   0),
		TO_BGR(0, 171,   0),
		TO_BGR(0, 147,  59),
		TO_BGR(0, 131, 139),
		TO_BGR(0,   0,   0),
		TO_BGR(0,   0,   0),
		TO_BGR(0,   0,   0),
		TO_BGR(255, 255, 255),
		TO_BGR(63, 191, 255),
		TO_BGR(95, 115, 255),
		TO_BGR(167, 139, 253),
		TO_BGR(247, 123, 255),
		TO_BGR(255, 119, 183),
		TO_BGR(255, 119,  99),
		TO_BGR(255, 155,  59),
		TO_BGR(243, 191,  63),
		TO_BGR(131, 211,  19),
		TO_BGR(79, 223,  75),
		TO_BGR(88, 248, 152),
		TO_BGR(0, 235, 219),
		TO_BGR(117, 117, 117),
		TO_BGR(0,   0,   0),
		TO_BGR(0,   0,   0),
		TO_BGR(255, 255, 255),
		TO_BGR(171, 231, 255),
		TO_BGR(199, 215, 255),
		TO_BGR(215, 203, 255),
		TO_BGR(255, 199, 255),
		TO_BGR(255, 199, 219),
		TO_BGR(255, 191, 179),
		TO_BGR(255, 219, 171),
		TO_BGR(255, 231, 163),
		TO_BGR(227, 255, 163),
		TO_BGR(171, 243, 191),
		TO_BGR(179, 255, 207),
		TO_BGR(159, 255, 243),
		TO_BGR(188, 188, 188),
		TO_BGR(0,   0,   0),
		TO_BGR(0,   0,   0),
	};
#undef TO_BGR

	//!< 2 プレーンに分けて出力、2 プレーンを合わせるとカラーインデックスが求まる
	//!< パターン 8 x 8 を表すのに
	//!<	最初の u8 x 8 へプレーン 0、続く u8 x 8 へプレーン 1
	//!<	u8[00] 00000000
	//!<	u8[01] 00000000
	//!<	....
	//!<	u8[14] 11111111
	//!<	u8[15] 11111111
	template<uint8_t W, uint8_t H>
	class ConverterBase : public Converter<W, H>
	{
	private:
		using Super = Converter<W, H>;
	public:
		ConverterBase(const cv::Mat& Img) : Super(Img) {}

		//!< 一番近い色のインデックスを返す
		virtual uint16_t ToPlatformColor(const cv::Vec3b& Color) const override {
			uint8_t Index = 0xff;
			float minDistSq = std::numeric_limits<float>::max();
			for (auto i = 0; i < size(ColorEntries); ++i) {
				const auto d = cv::Vec3f(ColorEntries[i]) - cv::Vec3f(Color);
				const auto distSq = d.dot(d);
				if (distSq < minDistSq) {
					minDistSq = distSq;
					Index = i;
				}
			}
			return Index;
		}
		virtual cv::Vec3b FromPlatformColor(const uint16_t& Index) const override {
			if (Index < size(ColorEntries)) {
				return ColorEntries[Index];
			}
			return cv::Vec3b(0, 0, 0);
		}

		virtual uint16_t GetPaletteCount() const override { return 4; };
		virtual uint16_t GetPaletteColorCount() const override { return 4; }

		virtual ConverterBase& CreatePattern() override {
			Super::CreatePattern();
			assert(size(this->Patterns) <= 256);
			return *this;
		}

		virtual const ConverterBase& OutputPalette(std::string_view Name) const override {
			this->OutputPaletteOfType<uint8_t>(Name);
			return *this;
		}
		virtual const ConverterBase& OutputPattern(std::string_view Name) const override {
			std::cout << "\tPattern count = " << size(this->Patterns) << std::endl;
			std::cout << "\tSprite size = " << static_cast<uint16_t>(W) << " x " << static_cast<uint16_t>(H) << std::endl;

			std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
			assert(!OutBin.bad());
			std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
			assert(!OutText.bad());

			//OutText << "const " << typeid(uint8_t).name() << " " << Name << "[] = {" << std::endl;
			OutText << "const u" << (sizeof(uint8_t) << 3) << " " << Name << "[] = {" << std::endl;

			for (auto pat = 0; pat < size(this->Patterns); ++pat) {
				const auto& Pat = this->Patterns[pat];
				assert(Pat.HasValidPaletteIndex());

				//!< パターン毎のパレットインデックス情報を出力
				std::cout << "\t\tPalette index = " << Pat.PaletteIndex << std::endl;

				//!< 2 プレーン
				for (auto pl = 0; pl < 2; ++pl) {
					OutText << "\t";
					for (auto i = 0; i < size(Pat.ColorIndices); ++i) {
						uint8_t Plane = 0;
						for (auto j = 0; j < size(Pat.ColorIndices[i]); ++j) {
							const auto ColorIndex = Pat.ColorIndices[i][j] + this->GetPaletteReservedColorCount(); //!< 先頭の透明色を考慮
							const auto Shift = 7 - j;
							const auto Mask = 1 << pl;
							Plane |= ((ColorIndex & Mask) ? 1 : 0) << Shift;
						}
						OutText << "0x" << std::hex << std::setw(sizeof(Plane) << 1) << std::right << std::setfill('0') << static_cast<uint16_t>(Plane);
						if (size(this->Patterns) - 1 > pat || 1 > pl || size(Pat.ColorIndices) - 1 > i) { OutText << ", "; }

						OutBin.write(reinterpret_cast<const char*>(&Plane), sizeof(Plane));
					}
				}
				OutText << std::endl;
			}
			OutText << "};" << std::endl;

			OutBin.close();
			OutText.close();

			return *this;
		}
	};

	namespace BG
	{
		//!< BG
		//!< 2 プレーンに分けて出力、2 プレーンを合わせるとカラーインデックスが求まる
		//!< アトリビュート
		//!<	パレット番号はセル毎に持つことはできず、2 x 2 セルでまとめて 1 パレット番号となる
		//!<	1 つの u8 で 4 x 4 セルを表し、各々の 2 ビットが 2 x 2 セルを表す
		//!<	(最初の 2 ビットが LT、続いて RT、LB、RB)
		//!<	01 23
		//!<	45 67
		template<uint8_t W = 8, uint8_t H = 8>
		class Converter : public ConverterBase<W, H>
		{
		private:
			using Super = ConverterBase<W, H>;
		public:
			Converter(const cv::Mat& Img) : Super(Img) {}

			virtual Converter& Create() override { Super::Create(); return *this; }

			virtual const Converter& OutputBAT(std::string_view Name) const override {
				std::cout << "\tBAT size = " << size(this->Map[0]) << " x " << size(this->Map) << std::endl;

				std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
				assert(!OutBin.bad());
				std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
				assert(!OutText.bad());

				//OutText << "const " << typeid(uint8_t).name() << " " << Name << "[] = {" << std::endl;
				OutText << "const u" << (sizeof(uint8_t) << 3) << " " << Name << "[] = {" << std::endl;

				//!< 4 x 4 分を 1 つの uint8_t で指定
				for (auto i = 0; i < size(this->Map); i += 4) {
					for (auto j = 0; j < size(this->Map[i]); j += 4) {
						//!< 2 x 2 分を uint8_t の 2 ビットで指定 (この 2 x 2 分は同じパレット番号でないといけない)
						const auto LTLT = static_cast<uint8_t>(this->Map[i + 0][j + 0].PatternIndex);
						const auto LTRT = static_cast<uint8_t>(this->Map[i + 0][j + 1].PatternIndex);
						const auto LTLB = static_cast<uint8_t>(this->Map[i + 1][j + 0].PatternIndex);
						const auto LTRB = static_cast<uint8_t>(this->Map[i + 1][j + 1].PatternIndex);
						//!< 2 x 2 分が同じパレット番号になっていない場合 assert
						assert(this->Patterns[LTLT].PaletteIndex == this->Patterns[LTRT].PaletteIndex == this->Patterns[LTLB].PaletteIndex == this->Patterns[LTRB].PaletteIndex);

						const auto RTLT = static_cast<uint8_t>(this->Map[i + 0][j + 2].PatternIndex);
						const auto RTRT = static_cast<uint8_t>(this->Map[i + 0][j + 3].PatternIndex);
						const auto RTLB = static_cast<uint8_t>(this->Map[i + 1][j + 2].PatternIndex);
						const auto RTRB = static_cast<uint8_t>(this->Map[i + 1][j + 3].PatternIndex);
						assert(this->Patterns[RTLT].PaletteIndex == this->Patterns[RTRT].PaletteIndex == this->Patterns[RTLB].PaletteIndex == this->Patterns[RTRB].PaletteIndex);

						const auto LBLT = static_cast<uint8_t>(this->Map[i + 2][j + 0].PatternIndex);
						const auto LBRT = static_cast<uint8_t>(this->Map[i + 2][j + 1].PatternIndex);
						const auto LBLB = static_cast<uint8_t>(this->Map[i + 3][j + 0].PatternIndex);
						const auto LBRB = static_cast<uint8_t>(this->Map[i + 3][j + 1].PatternIndex);
						assert(this->Patterns[LBLT].PaletteIndex == this->Patterns[LBRT].PaletteIndex == this->Patterns[LBLB].PaletteIndex == this->Patterns[LBRB].PaletteIndex);

						const auto RBLT = static_cast<uint8_t>(this->Map[i + 2][j + 2].PatternIndex);
						const auto RBRT = static_cast<uint8_t>(this->Map[i + 2][j + 3].PatternIndex);
						const auto RBLB = static_cast<uint8_t>(this->Map[i + 3][j + 2].PatternIndex);
						const auto RBRB = static_cast<uint8_t>(this->Map[i + 3][j + 3].PatternIndex);
						assert(this->Patterns[RBLT].PaletteIndex == this->Patterns[RBRT].PaletteIndex == this->Patterns[RBLB].PaletteIndex == this->Patterns[RBRB].PaletteIndex);

						assert(this->Patterns[RBLT].HasValidPaletteIndex());
						assert(this->Patterns[LBLT].HasValidPaletteIndex());
						assert(this->Patterns[RTLT].HasValidPaletteIndex());
						assert(this->Patterns[LTLT].HasValidPaletteIndex());
						const uint8_t BAT = (this->Patterns[RBLT].PaletteIndex << 6) | (this->Patterns[LBLT].PaletteIndex << 4) | (this->Patterns[RTLT].PaletteIndex << 2) | this->Patterns[LTLT].PaletteIndex;

						OutText << "0x" << std::hex << std::setw(sizeof(BAT) << 1) << std::right << std::setfill('0') << static_cast<uint16_t>(BAT);
						if (size(this->Map) - 1 > i || size(this->Map[i]) - 1 > j) { OutText << ", "; }

						OutBin.write(reinterpret_cast<const char*>(&BAT), sizeof(BAT));
					}
					OutText << std::endl;
				}
				OutText << "};" << std::endl;

				OutBin.close();
				OutText.close();

				return *this;
			}
		};
	}
	namespace Sprite
	{
		//!< スプライト
		//!< 2 プレーンに分けて出力、2 プレーンを合わせるとカラーインデックスが求まる
		template<uint8_t W = 8, uint8_t H = 8>
		class Converter : public ConverterBase<W, H>
		{
		private:
			using Super = ConverterBase<W, H>;
		public:
			Converter(const cv::Mat& Img) : Super(Img) {}

			virtual Converter& Create() override { Super::Create(); return *this; }
		};
	}

	class ResourceReader : public ResourceReaderBase
	{
	private:
		using Super = ResourceReaderBase;
	public:
		virtual void ProcessPalette(std::string_view Name, std::string_view File) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Palette ] " << Name << " (" << File << ")" << std::endl;

				BG::Converter<>(Image).Create().OutputPalette(Name).RestorePalette();
			}
		}
		virtual void ProcessTileSet(std::string_view Name, std::string_view File, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] std::string_view Option) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Pattern ] " << Name << " (" << File << ")" << std::endl;

				BG::Converter<>(Image).Create().OutputPattern(Name).RestorePattern();
			}
		}
		virtual void ProcessMap(std::string_view Name, std::string_view File, std::string_view TileSet, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Mapbase) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));

				std::cout << "[ Output BAT ] " << Name << " (" << File << ")" << std::endl;
				BG::Converter<>(Image).Create().OutputBAT(Name).RestoreMap();
			}
		}
		virtual void ProcessSprite(std::string_view Name, std::string_view File, const uint32_t Width, const uint32_t Height, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Time, [[maybe_unused]] std::string_view Collision, [[maybe_unused]] std::string_view Option, [[maybe_unused]] const uint32_t Iteration) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Sprite ] " << Name << " (" << File << ")" << std::endl;

				//!< 8x8 or 8x16
				switch (Width << 3)
				{
				case 8:
					switch (Height << 3) {
					case 8:
						Sprite::Converter<8, 8>(Image).Create().OutputPattern(Name).OutputAnimation(Name).RestorePattern();
						break;
					case 16:
						Sprite::Converter<8, 16>(Image).Create().OutputPattern(Name).OutputAnimation(Name).RestorePattern();
						break;
					default:
						std::cerr << "Sprite size not supported" << std::endl;
						break;
					}
					break;
				default:
					std::cerr << "Sprite size not supported" << std::endl;
					break;
				}
			}
		}
	};
}
#pragma endregion //!< FC

#pragma region GB
/*
* 画面		: 160 x 144
* BG		: 256 x 256(32 x 32 セル)、パターン 8 x 8 x 128 種(+ 共用分が 128 種ある)、4 諧調モノクロ
* スプライト	: 40 枚、パターン 8 x 8 x 128 種(+ 共用分が 128 種ある)。4 諧調モノクロ
*/
namespace GB
{
#define TO_BGR(r, g, b) cv::Vec3b(b, g, r)
	static const std::array ColorEntries = {
		TO_BGR(15, 56, 15),
		TO_BGR(48, 98, 48),
		TO_BGR(139, 172, 15),
		TO_BGR(155, 188, 15),
	};
#undef TO_BGR
	template<uint8_t W, uint8_t H>
	class ConverterBase : public Converter<W, H>
	{
	private:
		using Super = Converter<W, H>;
	public:
		ConverterBase(const cv::Mat& Img) : Super(Img) {}

		virtual uint16_t ToPlatformColor(const cv::Vec3b& Color) const override {
			uint8_t Index = 0xff;
			float minDistSq = std::numeric_limits<float>::max();
			for (auto i = 0; i < size(ColorEntries); ++i) {
				const auto d = cv::Vec3f(ColorEntries[i]) - cv::Vec3f(Color);
				const auto distSq = d.dot(d);
				if (distSq < minDistSq) {
					minDistSq = distSq;
					Index = i;
				}
			}
			return Index;
		}
		virtual cv::Vec3b FromPlatformColor(const uint16_t& Index) const override {
			if (Index < size(ColorEntries)) {
				return ColorEntries[Index];
			}
			return cv::Vec3b(0, 0, 0);
		}

		virtual uint16_t GetPaletteCount() const override { return 1; };
		virtual uint16_t GetPaletteColorCount() const override { return 4; }

		virtual ConverterBase& CreatePattern() override {
			Super::CreatePattern();

			if (size(this->Patterns) > 128) {
				//!< 共用分のパターン領域を使用する必要がある
				std::cerr << "Pattern count = " << size(this->Patterns) << " > 128" << std::endl;
			}
			//!< 共用分のパターン領域を使用しても足りない
			assert(size(this->Patterns) <= 256);
			return *this;
		}

		virtual const ConverterBase& OutputPalette(std::string_view Name) const override {
			std::cout << "\tPalette count = " << size(this->Palettes) << " / " << GetPaletteCount() << (size(this->Palettes) > GetPaletteCount() ? " warning" : "") << std::endl;

			std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
			assert(!OutBin.bad());
			std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
			assert(!OutText.bad());

			OutText << "const u" << (sizeof(uint8_t) << 3) << " " << Name << "[] = {" << std::endl;

			for (auto i = 0; i < size(this->Palettes); ++i) {
				const auto MaxCount = this->GetPaletteColorCount() - this->GetPaletteReservedColorCount();
				std::cout << "\t\tPalette color count = " << size(this->Palettes[i]) << " / " << MaxCount << (size(this->Palettes[i]) > MaxCount ? " warning" : "") << std::endl;

				const uint8_t TransparentColor = 0; //!< 先頭色 (ここでは 0 としている)

				//!< 出力用の型へ変換
				std::vector<uint8_t> PalOut;
				{
					if (this->HasPaletteReservedColor()) {
						PalOut.emplace_back(TransparentColor);
					}
					std::ranges::copy(this->Palettes[i], std::back_inserter(PalOut));
					for (auto j = size(PalOut); j < GetPaletteColorCount(); j++) {
						PalOut.emplace_back(TransparentColor);
					}
				}

				//!< 出力
				OutText << "\t";
				uint8_t PalMask = 0;
				for (auto j = 0; j < size(PalOut); ++j) {
					PalMask |= static_cast<uint16_t>(PalOut[j]) << (j << 1);
				}
				OutText << "0x" << std::hex << std::setw(sizeof(PalMask) << 1) << std::right << std::setfill('0') << static_cast<uint16_t>(PalMask);
				if (size(this->Palettes) - 1 > i) { OutText << ", "; }
				OutText << std::endl;

				OutBin.write(reinterpret_cast<const char*>(&PalMask), sizeof(PalMask));
			}
			OutText << "};" << std::endl;

			OutBin.close();
			OutText.close();

			return *this;
		}
		virtual const ConverterBase& OutputPattern(std::string_view Name) const override {
			std::cout << "\tPattern count = " << size(this->Patterns) << std::endl;
			std::cout << "\tSprite size = " << static_cast<uint16_t>(W) << " x " << static_cast<uint16_t>(H) << std::endl;

			std::ofstream OutBin(data(std::string(Name) + ".bin"), std::ios::binary | std::ios::out);
			assert(!OutBin.bad());
			std::ofstream OutText(data(std::string(Name) + ".txt"), std::ios::out);
			assert(!OutText.bad());

			//OutText << "const " << typeid(uint8_t).name() << " " << Name << "[] = {" << std::endl;
			OutText << "const u" << (sizeof(uint8_t) << 3) << " " << Name << "[] = {" << std::endl;

			for (auto pat = 0; pat < size(this->Patterns); ++pat) {
				const auto& Pat = this->Patterns[pat];
				assert(Pat.HasValidPaletteIndex());

				//!< パターン毎のパレットインデックス情報を出力
				std::cout << "\t\tPalette index = " << Pat.PaletteIndex << std::endl;

				OutText << "\t";
				for (auto i = 0; i < size(Pat.ColorIndices); ++i) {
					//!< 2 プレーン (GB ではプレーンをまとめて出力ではなく、交互に出力)
					for (auto pl = 0; pl < 2; ++pl) {
						uint8_t Plane = 0;
						for (auto j = 0; j < size(Pat.ColorIndices[i]); ++j) {
							const auto ColorIndex = Pat.ColorIndices[i][j] + this->GetPaletteReservedColorCount(); //!< 先頭の透明色を考慮
							const auto Shift = 7 - j;
							const auto Mask = 1 << pl;
							Plane |= ((ColorIndex & Mask) ? 1 : 0) << Shift;
						}
						OutText << "0x" << std::hex << std::setw(sizeof(Plane) << 1) << std::right << std::setfill('0') << static_cast<uint16_t>(Plane);
						if (size(this->Patterns) - 1 > pat || 1 > pl || size(Pat.ColorIndices) - 1 > i) { OutText << ", "; }

						OutBin.write(reinterpret_cast<const char*>(&Plane), sizeof(Plane));
					}
				}
				OutText << std::endl;
			}
			OutText << "};" << std::endl;

			OutBin.close();
			OutText.close();

			return *this;
		}
	};
	namespace BG
	{
		template<uint8_t W = 8, uint8_t H = 8>
		class Converter : public ConverterBase<W, H>
		{
		private:
			using Super = ConverterBase<W, H>;
		public:
			Converter(const cv::Mat& Img) : Super(Img) {}

			//!< GB の BG パレットは先頭が背景色というわけではない
			virtual bool HasPaletteReservedColor() const override { return false; }

			virtual Converter& Create() override { Super::Create(); return *this; }
		};
	}
	namespace Sprite
	{
		template<uint8_t W = 8, uint8_t H = 8>
		class Converter : public ConverterBase<W, H>
		{
		private:
			using Super = ConverterBase<W, H>;
		public:
			Converter(const cv::Mat& Img) : Super(Img) {}

			virtual uint16_t GetPaletteCount() const override { return 2; };

			virtual Converter& Create() override { Super::Create(); return *this; }
		};
	}

	class ResourceReader : public ResourceReaderBase
	{
	private:
		using Super = ResourceReaderBase;
	public:
		virtual void ProcessPalette(std::string_view Name, std::string_view File) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Palette ] " << Name << " (" << File << ")" << std::endl;

				BG::Converter<>(Image).Create().OutputPalette(Name).RestorePalette();
			}
		}
		virtual void ProcessTileSet(std::string_view Name, std::string_view File, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] std::string_view Option) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Pattern ] " << Name << " (" << File << ")" << std::endl;

				BG::Converter<>(Image).Create().OutputPattern(Name).RestorePattern();
			}
		}
		virtual void ProcessMap(std::string_view Name, std::string_view File, std::string_view TileSet, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Mapbase) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Map ] " << Name << " (" << File << ")" << std::endl;

				BG::Converter<>(Image).Create().OutputMap(Name).RestoreMap();
			}
		}
		virtual void ProcessSprite(std::string_view Name, std::string_view File, const uint32_t Width, const uint32_t Height, [[maybe_unused]] std::string_view Compression, [[maybe_unused]] const uint32_t Time, [[maybe_unused]] std::string_view Collision, [[maybe_unused]] std::string_view Option, [[maybe_unused]] const uint32_t Iteration) override {
			if (!empty(File)) {
				auto Image = cv::imread(data(File));
				std::cout << "[ Output Sprite ] " << Name << " (" << File << ")" << std::endl;

				//!< 8x8 or 8x16
				switch (Width << 3)
				{
				case 8:
					switch (Height << 3) {
					case 8:
						Sprite::Converter<8, 8>(Image).Create().OutputPattern(Name).OutputAnimation(Name).RestorePattern();
						break;
					case 16:
						Sprite::Converter<8, 16>(Image).Create().OutputPattern(Name).OutputAnimation(Name).RestorePattern();
						break;
					default:
						std::cerr << "Sprite size not supported" << std::endl;
						break;
					}
					break;
				default:
					std::cerr << "Sprite size not supported" << std::endl;
					break;
				}
			}
		}
	};
}
#pragma endregion //!< GB

int main(const int argc, const char *argv[])
{
	enum PLATFORM {
		PCE,
		FC,
		GB,
		GBC,
	};
#ifdef _DEBUG
	std::string Path = ".\\resPCE";
	auto Platform = PCE;
	//std::string Path = ".\\resFC";
	//auto Platform = FC;
	//std::string Path = ".\\resGB";
	//auto Platform = GB;
	//std::string Path = ".\\resGBC";
	//auto Platform = GBC;
#else
	std::string Path = ".";
	auto Platform = PCE;
#endif

	if (2 < argc) {
		Path = argv[2];
	}
	std::cout << Path << std::endl;
	if (1 < argc) {
		std::string Option;
		std::ranges::transform(std::string_view(argv[1]), std::back_inserter(Option), [](const char rhs) { return std::toupper(rhs, std::locale("")); });

		//!< C++23 なら contains() が使えるみたい
		if (std::string_view::npos != Option.find("PCE")) {
			Platform = PCE;
		}
		else if (std::string_view::npos != Option.find("FC")) {
			Platform = FC;
			FC::ResourceReader rr;
			rr.Read(Path);
		}
		else if (std::string_view::npos != Option.find("GBC") || std::string_view::npos != Option.find("CGB")) {
			Platform = GBC;
			//GBC::ResourceReader rr;
			//rr.Read(Path);
		}
		else if (std::string_view::npos != Option.find("GB")) {
			Platform = GB;
			GB::ResourceReader rr;
			rr.Read(Path);
		}
		else if (std::string_view::npos != Option.find("HELP")) {
			std::cout << "Usage : " << std::filesystem::path(argv[0]).filename().string() << " " << "[Platform]" << " " << "[Resource folder]" << std::endl;
			std::cout << "\tPlatform : PCE, FC, GB, CGB(GBC)" << std::endl;

			return 0;
		}
	}
	switch (Platform) {
	case PCE:
	{
		std::cout << "Platform : PCE" << std::endl;
		PCE::ResourceReader rr;
		rr.Read(Path);
	}
		break;
	case FC:
	{
		std::cout << "Platform : FC" << std::endl;
		FC::ResourceReader rr;
		rr.Read(Path);
	}
	break;
	case GB:
	{
		std::cout << "Platform : GB" << std::endl;
		GB::ResourceReader rr;
		rr.Read(Path);
	}
	break;
	case GBC:
	{
		std::cout << "Platform : CGB(GBC)" << std::endl;
		//GBC::ResourceReader rr;
		//rr.Read(Path);
	}
	break;
	default:
		break;
	}
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
